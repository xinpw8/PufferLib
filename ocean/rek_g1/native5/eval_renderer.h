#pragma once

// Presentation only. No mj_step calls: the training runtime owns simulation.
#include <mujoco/mujoco.h>
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <zlib.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace rek_eval {
inline std::string base64(const unsigned char* data, size_t length) {
    const char* chars="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out; out.reserve((length+2)/3*4);
    for(size_t i=0;i<length;i+=3) {
        uint32_t n=uint32_t(data[i])<<16;
        if(i+1<length)n|=uint32_t(data[i+1])<<8;
        if(i+2<length)n|=data[i+2];
        out+=chars[n>>18];out+=chars[(n>>12)&63];
        out+=i+1<length?chars[(n>>6)&63]:'=';
        out+=i+2<length?chars[n&63]:'=';
    }
    return out;
}
inline void u32(std::vector<unsigned char>& out,uint32_t x) {
    for(int i=3;i>=0;--i)out.push_back((x>>(i*8))&255);
}
inline void png_chunk(std::vector<unsigned char>& out,const char* type,
        const unsigned char* bytes,size_t count) {
    u32(out,uint32_t(count));size_t start=out.size();
    out.insert(out.end(),type,type+4);
    if(count)out.insert(out.end(),bytes,bytes+count);
    u32(out,uint32_t(crc32(0,out.data()+start,uInt(count+4))));
}
inline std::vector<unsigned char> png_rgb(const std::vector<unsigned char>& rgb,int w,int h) {
    std::vector<unsigned char> raw(size_t(h)*(size_t(w)*3+1));
    for(int y=0;y<h;y++) {
        size_t offset=size_t(y)*(w*3+1);raw[offset]=0;
        memcpy(raw.data()+offset+1,rgb.data()+size_t(h-1-y)*w*3,size_t(w)*3);
    }
    uLongf n=compressBound(raw.size());std::vector<unsigned char> compressed(n);
    if(compress2(compressed.data(),&n,raw.data(),raw.size(),1)!=Z_OK)
        throw std::runtime_error("PNG compression failed");
    std::vector<unsigned char> out={137,80,78,71,13,10,26,10},header;
    u32(header,w);u32(header,h);header.insert(header.end(),{8,2,0,0,0});
    png_chunk(out,"IHDR",header.data(),header.size());
    png_chunk(out,"IDAT",compressed.data(),n);png_chunk(out,"IEND",nullptr,0);
    return out;
}

class Renderer {
    EGLDisplay display=EGL_NO_DISPLAY;
    EGLContext context=EGL_NO_CONTEXT;
    EGLSurface surface=EGL_NO_SURFACE;
    mjModel* model=nullptr;
    mjData* data=nullptr;
    mjvScene scene{};
    mjrContext render_context{};
    mjvOption option{};
    mjvCamera camera{};
    int width=640,height=360;
public:
    explicit Renderer(const char* path) {
        char error[2048]{};model=mj_loadXML(path,nullptr,error,sizeof(error));
        if(!model)throw std::runtime_error(error);
        if(model->nq!=72)throw std::runtime_error("Renderer requires the two-fighter 72-qpos model");
        for(int g=0;g<model->ngeom;g++) {
            const char* name=mj_id2name(model,mjOBJ_GEOM,g);if(!name)continue;
            float* rgba=model->geom_rgba+g*4;
            if(std::string(name).rfind("arena_Collider_Wall_",0)==0 ||
               std::string(name).rfind("arena_Collider_Pillar_",0)==0)rgba[3]=0;
            if(strncmp(name,"player__",8)==0){rgba[0]=.12f;rgba[1]=.42f;rgba[2]=.95f;rgba[3]=1;}
            if(strncmp(name,"opponent__",10)==0){rgba[0]=1;rgba[1]=.32f;rgba[2]=.06f;rgba[3]=1;}
        }
        auto query=(PFNEGLQUERYDEVICESEXTPROC)eglGetProcAddress("eglQueryDevicesEXT");
        auto platform=(PFNEGLGETPLATFORMDISPLAYEXTPROC)eglGetProcAddress("eglGetPlatformDisplayEXT");
        EGLDeviceEXT devices[16];EGLint count=0;
        if(query&&platform&&query(16,devices,&count))
            for(int i=0;i<count;i++) {
                EGLDisplay candidate=platform(EGL_PLATFORM_DEVICE_EXT,devices[i],nullptr);
                if(candidate!=EGL_NO_DISPLAY&&eglInitialize(candidate,nullptr,nullptr)){
                    display=candidate;break;
                }
            }
        if(display==EGL_NO_DISPLAY){display=eglGetDisplay(EGL_DEFAULT_DISPLAY);
            if(!eglInitialize(display,nullptr,nullptr))throw std::runtime_error("EGL initialize failed");}
        if(!eglBindAPI(EGL_OPENGL_API))throw std::runtime_error("EGL OpenGL unavailable");
        EGLint attrs[]={EGL_SURFACE_TYPE,EGL_PBUFFER_BIT,EGL_RENDERABLE_TYPE,EGL_OPENGL_BIT,
            EGL_RED_SIZE,8,EGL_GREEN_SIZE,8,EGL_BLUE_SIZE,8,EGL_DEPTH_SIZE,24,EGL_NONE};
        EGLConfig config;EGLint configs=0;
        if(!eglChooseConfig(display,attrs,&config,1,&configs)||configs!=1)
            throw std::runtime_error("EGL framebuffer unavailable");
        EGLint pbuffer[]={EGL_WIDTH,width,EGL_HEIGHT,height,EGL_NONE};
        surface=eglCreatePbufferSurface(display,config,pbuffer);
        context=eglCreateContext(display,config,EGL_NO_CONTEXT,nullptr);
        if(surface==EGL_NO_SURFACE||context==EGL_NO_CONTEXT||!eglMakeCurrent(display,surface,surface,context))
            throw std::runtime_error("EGL context unavailable");
        data=mj_makeData(model);mjv_defaultScene(&scene);mjr_defaultContext(&render_context);
        mjv_defaultOption(&option);mjv_defaultCamera(&camera);
        mjv_makeScene(model,&scene,2000);mjr_makeContext(model,&render_context,mjFONTSCALE_100);
        mjr_setBuffer(mjFB_OFFSCREEN,&render_context);
        camera.type=mjCAMERA_FREE;camera.azimuth=90;camera.elevation=-45;
    }
    std::string frame(const float* qpos) {
        for(int k=0;k<72;k++)data->qpos[k]=qpos[k];
        mju_zero(data->qvel,model->nv);
        // Forward kinematics is exclusively for rendering a state snapshot.
        mj_kinematics(model,data);mj_comPos(model,data);
        const double dx=qpos[0]-qpos[36],dy=qpos[1]-qpos[37];
        camera.lookat[0]=.5*(qpos[0]+qpos[36]);camera.lookat[1]=.5*(qpos[1]+qpos[37]);
        camera.lookat[2]=std::max(.9,.5*double(qpos[2]+qpos[38]));
        camera.distance=std::max(3.8,2.5+1.2*std::sqrt(dx*dx+dy*dy));
        mjv_updateScene(model,data,&option,nullptr,&camera,mjCAT_ALL,&scene);
        mjrRect viewport{0,0,width,height};mjr_render(viewport,&scene,&render_context);
        std::vector<unsigned char> rgb(size_t(width)*height*3);
        mjr_readPixels(rgb.data(),nullptr,viewport,&render_context);
        auto png=png_rgb(rgb,width,height);return base64(png.data(),png.size());
    }
    ~Renderer(){
        if(data)mj_deleteData(data);
        if(context!=EGL_NO_CONTEXT){mjr_freeContext(&render_context);mjv_freeScene(&scene);}
        if(model)mj_deleteModel(model);
        if(display!=EGL_NO_DISPLAY){eglMakeCurrent(display,EGL_NO_SURFACE,EGL_NO_SURFACE,EGL_NO_CONTEXT);
            if(context!=EGL_NO_CONTEXT)eglDestroyContext(display,context);
            if(surface!=EGL_NO_SURFACE)eglDestroySurface(display,surface);eglTerminate(display);}
    }
};
}
