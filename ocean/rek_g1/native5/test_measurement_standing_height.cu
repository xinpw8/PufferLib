// CPU FK reference and production calibration kernels. No CPU simulation step.
#include "measurement.cu"
#include "../../../vendor/cJSON.h"
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <limits>

static int checks=0;
static void require(bool value,const char* message){checks++;if(!value)throw std::runtime_error(message);}
static std::string read_file(const std::string& path){
    std::ifstream f(path,std::ios::binary);require(bool(f),"reference input unreadable");
    return std::string(std::istreambuf_iterator<char>(f),{});
}
using Json=std::unique_ptr<cJSON,decltype(&cJSON_Delete)>;
static Json json_file(const std::string& path){auto text=read_file(path);Json j(cJSON_Parse(text.c_str()),cJSON_Delete);require(bool(j),"reference JSON invalid");return j;}
static cJSON* member(cJSON* j,const char* key){auto p=cJSON_GetObjectItemCaseSensitive(j,key);require(p,"reference JSON member missing");return p;}
template<class T> static std::vector<T> get(const T* pointer,size_t count){
    std::vector<T> v(count);rek5::cuda_check(cudaMemcpy(v.data(),pointer,count*sizeof(T),cudaMemcpyDeviceToHost));return v;
}
template<class T> static void put(T* pointer,const std::vector<T>& v){rek5::cuda_check(cudaMemcpy(pointer,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice));}

struct Geometry {
    int roots[2],feet[2][8],floor;
    double radii[2][8];
    explicit Geometry(const mjModel* m){
        floor=mj_name2id(m,mjOBJ_GEOM,"arena_Collider_Floor_Rektagon");require(floor>=0,"floor missing");
        for(int s=0;s<2;s++){
            std::string prefix=s?"opponent__":"player__";
            roots[s]=mj_name2id(m,mjOBJ_BODY,(prefix+"pelvis_3266").c_str());require(roots[s]>=0,"pelvis missing");
            int n=0;
            for(const char* name:{"left_ankle_roll_link_3045","right_ankle_roll_link_3090"}){
                int b=mj_name2id(m,mjOBJ_BODY,(prefix+name).c_str()),count=0;require(b>=0,"tagged foot missing");
                for(int g=0;g<m->ngeom;g++)if(m->geom_bodyid[g]==b){
                    require(count<4&&m->geom_type[g]==mjGEOM_SPHERE,"foot sphere contract mismatch");
                    feet[s][n]=g;radii[s][n++]=m->geom_size[3*g];count++;
                }
                require(count==4,"foot geometry count mismatch");
            }
        }
    }
    // Native calculation: minimum double support per foot, Unity float result,
    // then minimum across feet and float pelvis subtraction.
    template<class T> float standing(const T* centers,const T* pelvis,int s)const{
        float low=std::numeric_limits<float>::max();
        for(int f=0;f<2;f++){
            double z=std::numeric_limits<double>::infinity();
            for(int k=4*f;k<4*f+4;k++)z=std::min(z,double(centers[3*feet[s][k]+2])-radii[s][k]);
            low=std::min(low,float(z));
        }
        return float(pelvis[3*roots[s]+2])-low;
    }
};

int main(int argc,char** argv){try{
    require(argc==4||(argc==5&&std::string(argv[4])=="--cpu-only"),"usage: test_measurement_standing_height MODEL EXPORT ASSETS [--cpu-only]");
    bool cpu_only=argc==5;char error[1024]{};
    std::unique_ptr<mjModel,decltype(&mj_deleteModel)> model(mj_loadXML(argv[1],nullptr,error,sizeof(error)),mj_deleteModel);
    require(bool(model),error);auto* m=model.get();require(m->nq==72&&m->nbody==63&&m->ngeom==91&&m->nu==58,"model dimensions mismatch");
    std::unique_ptr<mjData,decltype(&mj_deleteData)> data(mj_makeData(m),mj_deleteData);require(bool(data),"data allocation failed");
    Geometry geometry(m);auto export_json=json_file(argv[2]);auto manifest=json_file(std::string(argv[3])+"/semantic_duel_assets_manifest.json");
    require(member(export_json.get(),"runtime_timestep_seconds")->valuedouble==.002,"runtime timestep mismatch");
    m->opt.timestep=.002; // physics.cu build_frame_maps applies this same value.
    auto qpos0=member(export_json.get(),"model_qpos0");auto exported_initial=member(export_json.get(),"initial_qpos");
    require(cJSON_GetArraySize(qpos0)==72&&cJSON_GetArraySize(exported_initial)==72,"export pose shape mismatch");
    std::vector<float> initial(72);for(int k=0;k<72;k++){
        double q=cJSON_GetArrayItem(qpos0,k)->valuedouble;require(q==m->qpos0[k],"model qpos0/export mismatch");initial[k]=float(q);
    }
    cJSON* idle=nullptr;auto clips=member(manifest.get(),"clips");
    for(auto c=clips->child;c;c=c->next)if(std::string(member(c,"role")->valuestring)=="idle"){
        require(!idle,"multiple idle clips");idle=c;
    }
    require(idle,"idle clip missing");auto file=member(member(idle,"files"),"mujoco_joint_order");
    auto bytes=read_file(std::string(argv[3])+"/"+file->valuestring);require(bytes.size()>=29*sizeof(float),"idle frame truncated");
    float idle_positions[29];std::memcpy(idle_positions,bytes.data(),sizeof(idle_positions));
    auto hinges=member(export_json.get(),"hinges");int clipped=0;
    for(int side=0;side<2;side++)for(int j=0;j<29;j++){
        int i=side*29+j,ji=m->actuator_trnid[2*i];require(m->jnt_type[ji]==mjJNT_HINGE,"actuator not hinge");
        int address=m->jnt_qposadr[ji];require(address==member(cJSON_GetArrayItem(hinges,i),"qposadr")->valueint,"actuator order mismatch");
        float q=idle_positions[j];require(std::isfinite(q),"idle joint not finite");
        if(m->jnt_limited[ji])q=std::min(float(m->jnt_range[2*ji+1]),std::max(float(m->jnt_range[2*ji]),q));
        clipped+=q!=idle_positions[j];initial[address]=q;
    }
    for(int k=0;k<72;k++)require(initial[k]==float(cJSON_GetArrayItem(exported_initial,k)->valuedouble),"runtime idle reconstruction differs from exported initial pose");
    double max_translation_error=0,max_fk_cast_error=0,initial_native[2]{},initial_old[2]{};
    std::vector<float> base_centers,base_pelvis,base_mats,tilted_centers,tilted_pelvis,tilted_pose;
    for(int scenario=0;scenario<3;scenario++){
        mj_resetData(m,data.get());for(int k=0;k<72;k++)data->qpos[k]=initial[k];
        if(scenario==1)for(int s=0;s<2;s++)data->qpos[s*36+2]+=1;
        if(scenario==2)for(int s=0;s<2;s++){
            // Left-multiply each actual spawn orientation by a 0.3 rad roll.
            double* q=data->qpos+s*36+3;double a[4]={std::cos(.15),std::sin(.15),0,0},out[4];
            mju_mulQuat(out,a,q);std::copy(out,out+4,q);
        }
        mj_kinematics(m,data.get());
        std::vector<float> centers(data->geom_xpos,data->geom_xpos+91*3),pelvis(data->xpos,data->xpos+63*3),mats(data->geom_xmat,data->geom_xmat+91*9);
        for(int s=0;s<2;s++){
            float native=geometry.standing(data->geom_xpos,data->xpos,s),fp32=geometry.standing(centers.data(),pelvis.data(),s);
            double floor=data->geom_xpos[geometry.floor*3+2]+std::abs(data->geom_xmat[geometry.floor*9+8])*m->geom_size[geometry.floor*3+2];
            double old=double(pelvis[geometry.roots[s]*3+2])-float(floor);
            require(std::isfinite(native)&&native>0,"actual idle/tilted calibration invalid");
            max_fk_cast_error=std::max(max_fk_cast_error,std::abs(double(native)-fp32));
            if(scenario==0){initial_native[s]=native;initial_old[s]=old;}
            if(scenario==1){
                max_translation_error=std::max(max_translation_error,std::abs(double(native)-initial_native[s]));
                require(std::abs(double(native)-initial_native[s])<3e-7,"native denominator changed under translation");
                require(std::abs(old-initial_old[s]-1)<3e-7,"old floor negative control did not shift by one metre");
            }
        }
        if(scenario==0){base_centers=centers;base_pelvis=pelvis;base_mats=mats;}
        if(scenario==2){tilted_centers=centers;tilted_pelvis=pelvis;tilted_pose.assign(data->qpos,data->qpos+72);}
        std::printf("{\"event\":\"cpu_fk\",\"scenario\":%d,\"native_standing\":[%.9g,%.9g],\"old_floor_standing\":[%.9g,%.9g]}\n",scenario,
            geometry.standing(data->geom_xpos,data->xpos,0),geometry.standing(data->geom_xpos,data->xpos,1),
            double(pelvis[geometry.roots[0]*3+2])-(data->geom_xpos[geometry.floor*3+2]+m->geom_size[geometry.floor*3+2]),
            double(pelvis[geometry.roots[1]*3+2])-(data->geom_xpos[geometry.floor*3+2]+m->geom_size[geometry.floor*3+2]));
    }
    int gpu_cases=0,model_rejections=0;
    if(!cpu_only){
        rek5::DeviceStorage storage;rek5::Physics physics{};physics.model=m;
        auto& p=physics.data;p.arenas=1;p.bodies=63;p.geoms=91;p.capacity=1;
        p.qpos=storage.upload(initial);p.xpos=storage.upload(base_pelvis);p.geom_xpos=storage.upload(base_centers);p.geom_xmat=storage.upload(base_mats);
        std::unique_ptr<rek5::CombatMeasurement> measurement(rek5::measurement_create(&physics));
        auto view=rek5::fall_view(measurement.get());auto* d=static_cast<Measurement*>(measurement->descriptor);
        put(d->fall_valid,std::vector<uint8_t>{1});
        // Synthetic geometry cases preserve measured base pose unless stated.
        for(int mode=0;mode<11;mode++){
            auto centers=base_centers,pelvis=base_pelvis,pose=initial;auto local=view;
            if(mode==1)for(int s=0;s<2;s++){
                pelvis[geometry.roots[s]*3+2]+=1;
                for(int k=0;k<8;k++)centers[geometry.feet[s][k]*3+2]+=1;
            }
            if(mode==2)for(int s=0;s<2;s++)for(int k=0;k<4;k++)centers[geometry.feet[s][k]*3+2]+=.2f;
            if(mode==3)for(int s=0;s<2;s++)for(int k=0;k<4;k++){
                std::swap(local.foot_geoms[s][k],local.foot_geoms[s][7-k]);std::swap(local.foot_radii[s][k],local.foot_radii[s][7-k]);
            }
            // Explicit zero/negative/small-positive native disabled heights.
            if(mode>=4&&mode<=6)for(int s=0;s<2;s++){
                float low=pelvis[geometry.roots[s]*3+2]-geometry.standing(centers.data(),pelvis.data(),s);
                pelvis[geometry.roots[s]*3+2]=low+(mode==4?0.f:mode==5?-.01f:.00005f);
            }
            if(mode==7)centers[geometry.feet[0][0]*3+2]=std::numeric_limits<float>::quiet_NaN();
            if(mode==8)pose[3]=pose[4]=pose[5]=pose[6]=0;
            if(mode==9)pelvis[geometry.roots[0]*3+2]=std::numeric_limits<float>::infinity();
            if(mode==10){centers=tilted_centers;pelvis=tilted_pelvis;pose=tilted_pose;}
            put(p.xpos,pelvis);put(p.geom_xpos,centers);put(p.qpos,pose);rek5::cuda_check(cudaMemset(d->world_bad,0,sizeof(int32_t)));
            rek5::calibrate<<<1,32>>>(local);rek5::all_calibrated<<<1,32>>>(local);rek5::commit_calibrated<<<1,32>>>(local);
            rek5::sample_fall<<<1,32>>>(local,nullptr);rek5::cuda_check(cudaGetLastError());
            auto h=get(measurement->standing,2),f=get(measurement->fall_floats,10);auto valid=get(measurement->fall_valid,2),cal=get(measurement->calibrated,2);
            bool invalid=mode>=7&&mode<=9;
            for(int s=0;s<2;s++){
                require(bool(cal[s])==!invalid&&bool(valid[s])==!invalid,"global calibration invalid handling differs");
                if(invalid)continue;
                float expected=geometry.standing(centers.data(),pelvis.data(),s);
                require(h[s]==expected,"GPU support/subtraction differs from FP32-input native reference");
                float ratio=expected<=.0001f?1.f:(pelvis[geometry.roots[s]*3+2]-f[s*5+3])/expected;
                require(std::abs(f[s*5+1]-ratio)<1e-6f,"GPU disabled-height/ratio branch differs");
                if(mode>=4&&mode<=6)require(f[s*5+1]==1.f,"height fallback must be exactly one");
            }
            gpu_cases++;
        }
        put(measurement->standing,std::vector<float>{.0001f,std::nextafter(.0001f,1.f)});
        rek5::sample_fall<<<1,32>>>(view,nullptr);rek5::cuda_check(cudaGetLastError());
        auto boundary=get(measurement->fall_floats,10);
        require(boundary[1]==1.f,"exact disabled-height threshold differs");
        require(boundary[6]>100.f,"above-threshold height should use the division branch");gpu_cases++;
        // Constructor contract rejects unsupported/missing/nonfinite geometry.
        int g=geometry.feet[0][0],type=m->geom_type[g],body=m->geom_bodyid[g];double radius=m->geom_size[3*g];
        for(int mode=0;mode<3;mode++){
            if(mode==0)m->geom_type[g]=mjGEOM_BOX;
            if(mode==1)m->geom_size[3*g]=std::numeric_limits<double>::quiet_NaN();
            if(mode==2)m->geom_bodyid[g]=geometry.roots[0];
            bool rejected=false;try{std::unique_ptr<rek5::CombatMeasurement> bad(rek5::measurement_create(&physics));}catch(const std::runtime_error&){rejected=true;}
            require(rejected,"invalid foot geometry model accepted");model_rejections++;
            m->geom_type[g]=type;m->geom_bodyid[g]=body;m->geom_size[3*g]=radius;
        }
    }
    std::printf("{\"event\":\"standing_height_result\",\"passed\":true,\"cpu_only\":%s,\"checks\":%d,\"actual_runtime_idle_reconstructed\":true,\"clipped_joints\":%d,\"gpu_cases\":%d,\"model_rejections\":%d,\"max_translation_rounding_m\":%.17g,\"max_double_vs_fp32_fk_height_m\":%.17g,\"physics_steps\":0,\"controller_inference\":false}\n",cpu_only?"true":"false",checks,clipped,gpu_cases,model_rejections,max_translation_error,max_fk_cast_error);
    return 0;
}catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}}
