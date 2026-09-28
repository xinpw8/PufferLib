#include "eval_renderer.h"
#include <fstream>
#include <iostream>
int main(int argc,char**argv){
    try {
        if(argc!=4)throw std::runtime_error("usage: render_saved presentation.xml saved-qpos.f32 original.xml");
        float q[72];std::ifstream f(argv[2],std::ios::binary);f.read(reinterpret_cast<char*>(q),sizeof(q));
        if(f.gcount()!=sizeof(q)||f.peek()!=EOF)throw std::runtime_error("expected exactly72 float32 values");
        for(float v:q)if(!std::isfinite(v))throw std::runtime_error("nonfinite saved pose");
        char error[2048]{};auto a=mj_loadXML(argv[1],nullptr,error,sizeof(error));if(!a)throw std::runtime_error(error);
        auto b=mj_loadXML(argv[3],nullptr,error,sizeof(error));if(!b)throw std::runtime_error(error);
        if(a->nq!=b->nq||a->nv!=b->nv||a->nbody!=b->nbody||a->njnt!=b->njnt)throw std::runtime_error("kinematic layout mismatch");
        for(int i=1;i<a->nbody;i++)if(std::string(mj_id2name(a,mjOBJ_BODY,i))!=mj_id2name(b,mjOBJ_BODY,i)||a->body_parentid[i]!=b->body_parentid[i])throw std::runtime_error("body mapping mismatch");
        for(int j=0;j<a->njnt;j++)if(std::string(mj_id2name(a,mjOBJ_JOINT,j))!=mj_id2name(b,mjOBJ_JOINT,j)||a->jnt_bodyid[j]!=b->jnt_bodyid[j]||a->jnt_type[j]!=b->jnt_type[j]||a->jnt_qposadr[j]!=b->jnt_qposadr[j]||a->jnt_dofadr[j]!=b->jnt_dofadr[j])throw std::runtime_error("joint mapping mismatch");
        auto ad=mj_makeData(a);auto bd=mj_makeData(b);for(int i=0;i<72;i++)ad->qpos[i]=bd->qpos[i]=q[i];mj_kinematics(a,ad);mj_kinematics(b,bd);
        double pos=0,quat=0;for(int i=0;i<a->nbody*3;i++)pos=std::max(pos,std::abs(ad->xpos[i]-bd->xpos[i]));for(int i=0;i<a->nbody*4;i++)quat=std::max(quat,std::abs(ad->xquat[i]-bd->xquat[i]));
        if(pos>1e-12||quat>1e-12)throw std::runtime_error("saved-pose FK mismatch");
        std::cerr<<"{\"nq\":"<<a->nq<<",\"nv\":"<<a->nv<<",\"bodies\":"<<a->nbody<<",\"joints\":"<<a->njnt<<",\"max_xpos_error\":"<<pos<<",\"max_xquat_error\":"<<quat<<",\"simulation_steps\":0}"<<std::endl;
        mj_deleteData(ad);mj_deleteData(bd);mj_deleteModel(a);mj_deleteModel(b);
        rek_eval::Renderer renderer(argv[1]);std::cout<<renderer.frame(q)<<std::endl;
        return 0;
    }catch(const std::exception&e){std::cerr<<e.what()<<std::endl;return 2;}
}
