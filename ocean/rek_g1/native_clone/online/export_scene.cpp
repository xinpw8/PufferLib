// Export the compiled presentation model for browser rendering.
// Writes scene.json (kinematic tree, geoms, materials, lights), scene.bin
// (de-indexed mesh positions, normals and texture coordinates as float32) and
// tex_N.png holding MuJoCo's decoded texture bytes, so the browser uses exactly
// the pixels and coordinates MuJoCo renders with.
#include <mujoco/mujoco.h>
#include <zlib.h>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void u32(std::vector<unsigned char>& out,uint32_t x){for(int i=3;i>=0;--i)out.push_back((x>>(i*8))&255);}
void chunk(std::vector<unsigned char>& out,const char* type,const unsigned char* bytes,size_t count){
    u32(out,uint32_t(count));size_t start=out.size();out.insert(out.end(),type,type+4);
    if(count)out.insert(out.end(),bytes,bytes+count);
    u32(out,uint32_t(crc32(0,out.data()+start,uInt(count+4))));
}
// Rows are written in MuJoCo's order; the browser uploads them with flipY off.
std::vector<unsigned char> png(const unsigned char* data,int w,int h,int channels){
    if(channels!=3&&channels!=4)throw std::runtime_error("Unsupported texture channel count");
    std::vector<unsigned char> raw(size_t(h)*(size_t(w)*channels+1));
    for(int y=0;y<h;y++){size_t o=size_t(y)*(size_t(w)*channels+1);raw[o]=0;
        memcpy(raw.data()+o+1,data+size_t(y)*w*channels,size_t(w)*channels);}
    uLongf n=compressBound(raw.size());std::vector<unsigned char> z(n);
    if(compress2(z.data(),&n,raw.data(),raw.size(),9)!=Z_OK)throw std::runtime_error("PNG compression failed");
    std::vector<unsigned char> out={137,80,78,71,13,10,26,10},hdr;
    u32(hdr,w);u32(hdr,h);hdr.insert(hdr.end(),{8,uint8_t(channels==4?6:2),0,0,0});
    chunk(out,"IHDR",hdr.data(),hdr.size());chunk(out,"IDAT",z.data(),n);chunk(out,"IEND",nullptr,0);
    return out;
}
template<class T>void arr(std::ostringstream& s,const T* v,int n){
    s<<'[';for(int i=0;i<n;i++){if(i)s<<',';s<<double(v[i]);}s<<']';
}
std::string name(const mjModel* m,int adr){return adr<0?"":std::string(m->names+adr);}
std::string esc(const std::string& v){std::string o;for(char c:v){if(c=='"'||c=='\\')o+='\\';o+=c;}return o;}
}

int main(int argc,char** argv){
    if(argc!=3&&argc!=4){std::fprintf(stderr,"Usage: export_scene MODEL.xml OUTPUT_DIR [QPOS.f32 -> check.json]\n");return 2;}
    char error[2048]{};mjModel* m=mj_loadXML(argv[1],nullptr,error,sizeof(error));
    if(!m){std::fprintf(stderr,"%s\n",error);return 1;}
    const std::string out=argv[2];
    std::ostringstream s;s.precision(9);
    s<<"{\"schema\":\"rek.online.scene.v1\",\"nq\":"<<m->nq<<",\"qpos0\":";arr(s,m->qpos0,m->nq);
    s<<",\"fovy\":"<<m->vis.global.fovy<<",\"extent\":"<<m->stat.extent<<",\"center\":";arr(s,m->stat.center,3);
    s<<",\"headlight\":{\"ambient\":";arr(s,m->vis.headlight.ambient,3);
    s<<",\"diffuse\":";arr(s,m->vis.headlight.diffuse,3);s<<",\"specular\":";arr(s,m->vis.headlight.specular,3);
    s<<",\"active\":"<<m->vis.headlight.active<<"}";
    s<<",\"bodies\":[";
    for(int b=0;b<m->nbody;b++){if(b)s<<',';
        s<<"{\"name\":\""<<esc(name(m,m->name_bodyadr[b]))<<"\",\"parent\":"<<m->body_parentid[b]<<",\"pos\":";arr(s,m->body_pos+3*b,3);
        s<<",\"quat\":";arr(s,m->body_quat+4*b,4);s<<",\"jntadr\":"<<m->body_jntadr[b]<<",\"jntnum\":"<<m->body_jntnum[b]<<'}';}
    s<<"],\"joints\":[";
    for(int j=0;j<m->njnt;j++){if(j)s<<',';
        s<<"{\"type\":"<<m->jnt_type[j]<<",\"qposadr\":"<<m->jnt_qposadr[j]<<",\"body\":"<<m->jnt_bodyid[j]<<",\"pos\":";arr(s,m->jnt_pos+3*j,3);
        s<<",\"axis\":";arr(s,m->jnt_axis+3*j,3);s<<'}';}
    s<<"],\"geoms\":[";
    bool first=true;int visible=0;
    for(int g=0;g<m->ngeom;g++){
        // Match mjv_updateScene defaults: groups 0-2 shown, fully transparent geoms hidden.
        const int group=m->geom_group[g];
        const float* rgba=m->geom_matid[g]>=0?m->mat_rgba+4*m->geom_matid[g]:m->geom_rgba+4*g;
        if(group<0||group>2||rgba[3]==0)continue;
        if(!first)s<<',';
        first=false;visible++;
        s<<"{\"id\":"<<g<<",\"name\":\""<<esc(name(m,m->name_geomadr[g]))<<"\",\"body\":"<<m->geom_bodyid[g]<<",\"type\":"<<m->geom_type[g];
        s<<",\"size\":";arr(s,m->geom_size+3*g,3);s<<",\"pos\":";arr(s,m->geom_pos+3*g,3);
        s<<",\"quat\":";arr(s,m->geom_quat+4*g,4);s<<",\"rgba\":";arr(s,m->geom_rgba+4*g,4);
        s<<",\"mat\":"<<m->geom_matid[g]<<",\"mesh\":"<<(m->geom_type[g]==mjGEOM_MESH?m->geom_dataid[g]:-1)<<'}';
    }
    s<<"],\"materials\":[";
    for(int i=0;i<m->nmat;i++){if(i)s<<',';
        s<<"{\"rgba\":";arr(s,m->mat_rgba+4*i,4);s<<",\"tex\":"<<m->mat_texid[mjNTEXROLE*i+mjTEXROLE_RGB];
        s<<",\"specular\":"<<m->mat_specular[i]<<",\"shininess\":"<<m->mat_shininess[i]<<",\"emission\":"<<m->mat_emission[i];
        s<<",\"texrepeat\":";arr(s,m->mat_texrepeat+2*i,2);s<<",\"texuniform\":"<<int(m->mat_texuniform[i])<<'}';}
    s<<"],\"textures\":[";
    for(int t=0;t<m->ntex;t++){if(t)s<<',';
        const int w=m->tex_width[t],h=m->tex_height[t],c=m->tex_nchannel[t];
        auto bytes=png(m->tex_data+m->tex_adr[t],w,h,c);
        const std::string file="tex_"+std::to_string(t)+".png";
        std::ofstream(out+"/"+file,std::ios::binary).write((const char*)bytes.data(),bytes.size());
        s<<"{\"file\":\""<<file<<"\",\"type\":"<<m->tex_type[t]<<",\"width\":"<<w<<",\"height\":"<<h<<'}';}
    s<<"],\"lights\":[";
    for(int l=0;l<m->nlight;l++){if(l)s<<',';
        s<<"{\"body\":"<<m->light_bodyid[l]<<",\"pos\":";arr(s,m->light_pos+3*l,3);s<<",\"dir\":";arr(s,m->light_dir+3*l,3);
        s<<",\"diffuse\":";arr(s,m->light_diffuse+3*l,3);s<<",\"ambient\":";arr(s,m->light_ambient+3*l,3);
        s<<",\"specular\":";arr(s,m->light_specular+3*l,3);s<<",\"type\":"<<m->light_type[l]<<'}';}
    s<<"],\"meshes\":[";
    std::vector<float> bin;size_t faces=0;
    for(int k=0;k<m->nmesh;k++){if(k)s<<',';
        const int fadr=m->mesh_faceadr[k],fnum=m->mesh_facenum[k],vadr=m->mesh_vertadr[k],nadr=m->mesh_normaladr[k];
        const int tadr=m->mesh_texcoordadr[k];const bool uv=tadr>=0&&m->mesh_texcoordnum[k]>0;
        const size_t offset=bin.size();faces+=fnum;
        for(int f=0;f<fnum;f++)for(int c=0;c<3;c++){const float* p=m->mesh_vert+3*(vadr+m->mesh_face[3*(fadr+f)+c]);bin.insert(bin.end(),p,p+3);}
        for(int f=0;f<fnum;f++)for(int c=0;c<3;c++){const float* n=m->mesh_normal+3*(nadr+m->mesh_facenormal[3*(fadr+f)+c]);bin.insert(bin.end(),n,n+3);}
        if(uv)for(int f=0;f<fnum;f++)for(int c=0;c<3;c++){const float* t=m->mesh_texcoord+2*(tadr+m->mesh_facetexcoord[3*(fadr+f)+c]);bin.insert(bin.end(),t,t+2);}
        s<<"{\"offset\":"<<offset<<",\"vertices\":"<<3*fnum<<",\"uv\":"<<(uv?"true":"false")<<'}';
    }
    s<<"]}\n";
    std::ofstream(out+"/scene.json")<<s.str();
    std::ofstream(out+"/scene.bin",std::ios::binary).write((const char*)bin.data(),bin.size()*sizeof(float));
    std::printf("{\"bodies\":%d,\"joints\":%d,\"geoms\":%d,\"visible_geoms\":%d,\"meshes\":%d,\"faces\":%zu,\"bin_bytes\":%zu,\"textures\":%d,\"nq\":%d}\n",
        int(m->nbody),int(m->njnt),int(m->ngeom),visible,int(m->nmesh),faces,bin.size()*sizeof(float),int(m->ntex),int(m->nq));
    if(argc==4){
        // Reference poses for the browser kinematics test: MuJoCo geom frames for given qpos rows.
        std::ifstream in(argv[3],std::ios::binary|std::ios::ate);const size_t bytes=in.tellg();in.seekg(0);
        std::vector<float> q(bytes/sizeof(float));in.read((char*)q.data(),q.size()*sizeof(float));
        if(q.size()%m->nq)throw std::runtime_error("QPOS file is not a whole number of rows");
        mjData* d=mj_makeData(m);std::ostringstream c;c.precision(9);c<<"[";
        for(size_t r=0;r<q.size()/m->nq;r++){if(r)c<<',';
            for(int i=0;i<m->nq;i++)d->qpos[i]=q[r*m->nq+i];
            mj_kinematics(m,d);c<<"{\"xpos\":";arr(c,d->geom_xpos,3*m->ngeom);c<<",\"xmat\":";arr(c,d->geom_xmat,9*m->ngeom);c<<'}';}
        c<<"]\n";std::ofstream(out+"/check.json")<<c.str();mj_deleteData(d);
    }
    mj_deleteModel(m);return 0;
}
