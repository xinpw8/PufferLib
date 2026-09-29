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

// Third-person view behind one fighter. Offsets are the recovered
// RobotFollowCamera constructor defaults: 3 m behind, 1 m above and a
// 10-degree downward tilt. REK's scene overrides, active mode and FOV remain
// unverified, so this is a presentation choice, not a parity claim.
constexpr double kFollowBehindM=3.0,kFollowHeightM=1.0,kFollowPitchDeg=10.0;
// Pelvis yaw sways with gait and swings up to ~15 degrees during strikes. The
// view holds inside a dead band, trails turns at the band edge and recentres
// over simulated (tick) time, so wall-clock render pacing cannot change it.
constexpr double kFollowDeadbandDeg=20.0,kFollowRecentreSeconds=1.0,kControlSeconds=.02;
// A body forward axis this close to vertical (a downed fighter) has no heading.
constexpr double kFollowMinHeadingNorm=.2;
// Arena meshes whose wall plane the eye has crossed are dropped from the scene.
constexpr double kCutawayMarginM=.4,kCutawayMinRadiusM=1.0;

struct FollowView {
    int side=-1;
    bool has_tick=false,has_generation=false;
    double tick=0,generation=0;
};

inline double wrap_angle(double a){return std::atan2(std::sin(a),std::cos(a));}

// SonicPolicyRunner.CalcHeadingMj operand order (g1_heading_native.h), wxyz.
inline bool fighter_heading(const float* qpos,int side,double& out) {
    const float* q=qpos+36*side+3;
    const double numerator=2.0*(double(q[2])*q[1]+double(q[3])*q[0]);
    const double denominator=1.0-2.0*(double(q[3])*q[3]+double(q[2])*q[2]);
    if(std::hypot(numerator,denominator)<kFollowMinHeadingNorm)return false;
    out=std::atan2(numerator,denominator);return true;
}

class FollowHeading {
    bool valid=false;int side=-1;double heading=0,tick=0,generation=0;
public:
    double update(const float* qpos,const FollowView& view) {
        double target=0;const bool has_target=fighter_heading(qpos,view.side,target);
        const bool continuous=valid&&view.side==side&&view.has_tick&&view.has_generation
            &&view.generation==generation&&view.tick>=tick;
        if(!continuous) {
            if(!has_target) {
                const float* self=qpos+36*view.side;const float* other=qpos+36*(1-view.side);
                const double dx=other[0]-self[0],dy=other[1]-self[1];
                target=dx*dx+dy*dy>0?std::atan2(dy,dx):0;
            }
            heading=target;
        } else if(has_target) {
            const double band=kFollowDeadbandDeg*mjPI/180;
            double error=wrap_angle(target-heading);
            if(std::abs(error)>band){heading+=error-std::copysign(band,error);error=std::copysign(band,error);}
            const double seconds=(view.tick-tick)*kControlSeconds;
            heading=wrap_angle(heading+error*(1-std::exp(-seconds/kFollowRecentreSeconds)));
        }
        valid=true;side=view.side;tick=view.tick;generation=view.generation;
        return heading;
    }
};

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
    int width=1280,height=720;
    FollowHeading follow;
    struct CutawayGeom{int id;double ux,uy,radius;};
    std::vector<CutawayGeom> cutaway;
    std::vector<unsigned char> hidden;
    // Static world-body arena meshes, located by their world vertex centroid.
    void index_cutaway() {
        mj_kinematics(model,data);hidden.assign(model->ngeom,0);
        for(int g=0;g<model->ngeom;g++) {
            const int mesh=model->geom_dataid[g];
            if(model->geom_bodyid[g]!=0||model->geom_type[g]!=mjGEOM_MESH||mesh<0)continue;
            const int count=model->mesh_vertnum[mesh];if(count<=0)continue;
            const float* v=model->mesh_vert+3*model->mesh_vertadr[mesh];
            const mjtNum* R=data->geom_xmat+9*g;double sx=0,sy=0;
            for(int i=0;i<count;i++) {
                sx+=R[0]*v[3*i]+R[1]*v[3*i+1]+R[2]*v[3*i+2];
                sy+=R[3]*v[3*i]+R[4]*v[3*i+1]+R[5]*v[3*i+2];
            }
            const double cx=data->geom_xpos[3*g]+sx/count,cy=data->geom_xpos[3*g+1]+sy/count;
            const double radius=std::hypot(cx,cy);
            if(radius>=kCutawayMinRadiusM)cutaway.push_back({g,cx/radius,cy/radius,radius});
        }
    }
    void place_follow_camera(const float* qpos,const FollowView& view) {
        const double heading=follow.update(qpos,view),pitch=kFollowPitchDeg*mjPI/180;
        const float* p=qpos+36*view.side;
        camera.azimuth=heading*180/mjPI;camera.elevation=-kFollowPitchDeg;
        camera.distance=kFollowBehindM/std::cos(pitch);
        camera.lookat[0]=p[0];camera.lookat[1]=p[1];
        camera.lookat[2]=p[2]+kFollowHeightM-kFollowBehindM*std::tan(pitch);
        const double eye_x=p[0]-kFollowBehindM*std::cos(heading),eye_y=p[1]-kFollowBehindM*std::sin(heading);
        for(const auto& c:cutaway)hidden[c.id]=eye_x*c.ux+eye_y*c.uy>c.radius-kCutawayMarginM;
    }
    void drop_hidden_geoms() {
        int kept=0;
        for(int i=0;i<scene.ngeom;i++) {
            const mjvGeom& g=scene.geoms[i];
            if(g.objtype==mjOBJ_GEOM&&g.objid>=0&&g.objid<model->ngeom&&hidden[g.objid])continue;
            scene.geoms[kept++]=g;
        }
        scene.ngeom=kept;
    }
public:
    explicit Renderer(const char* path) {
        char error[2048]{};model=mj_loadXML(path,nullptr,error,sizeof(error));
        if(!model)throw std::runtime_error(error);
        if(model->nq!=72)throw std::runtime_error("Renderer requires the two-fighter 72-qpos model");
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
        camera.type=mjCAMERA_FREE;camera.azimuth=130;camera.elevation=-60;
        index_cutaway();
    }
    std::string frame(const float* qpos){return frame(qpos,FollowView{});}
    // view.side 0/1 follows that fighter; -1 keeps the two-fighter overview.
    std::string frame(const float* qpos,const FollowView& view) {
        if(view.side<-1||view.side>1)throw std::runtime_error("Renderer follow side must be -1, 0 or 1");
        for(int k=0;k<72;k++)data->qpos[k]=qpos[k];
        mju_zero(data->qvel,model->nv);
        // Forward kinematics is exclusively for rendering a state snapshot.
        mj_kinematics(model,data);mj_comPos(model,data);
        if(view.side>=0)place_follow_camera(qpos,view);
        else {
            const double dx=qpos[0]-qpos[36],dy=qpos[1]-qpos[37];
            camera.azimuth=130;camera.elevation=-60;
            camera.lookat[0]=.5*(qpos[0]+qpos[36]);camera.lookat[1]=.5*(qpos[1]+qpos[37]);
            camera.lookat[2]=std::max(.9,.5*double(qpos[2]+qpos[38]));
            camera.distance=std::max(3.8,2.5+1.2*std::sqrt(dx*dx+dy*dy));
        }
        mjv_updateScene(model,data,&option,nullptr,&camera,mjCAT_ALL,&scene);
        if(view.side>=0)drop_hidden_geoms();
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
