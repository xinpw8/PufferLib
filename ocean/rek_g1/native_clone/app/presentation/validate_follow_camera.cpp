// Offline checks for the third-person follow camera. No simulation steps.
// usage: validate_follow_camera presentation.xml saved-qpos.f32 OUTPUT_DIR
#include "eval_renderer.h"
#include "../../source/ocean/rek_g1/native5/eval_render_request.h"
#include <fstream>
#include <iostream>
using namespace rek_eval;

static int failures=0;
static void check(bool ok,const char* what){std::cout<<(ok?"PASS ":"FAIL ")<<what<<"\n";if(!ok)failures++;}
static void yaw(float* q,int side,double degrees){
    const double half=degrees*mjPI/360;float* r=q+36*side+3;
    r[0]=float(std::cos(half));r[1]=0;r[2]=0;r[3]=float(std::sin(half));
}
static void save(const std::string& b64,const std::string& path){
    static const std::string chars="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;int value=0,bits=-8;
    for(char c:b64){if(c=='=')break;value=(value<<6)+int(chars.find(c));bits+=6;
        if(bits>=0){out.push_back(char((value>>bits)&255));bits-=8;}}
    std::ofstream(path,std::ios::binary)<<out;
}
static bool parse_side(const std::string& extra,int& side){
    std::string text="{\"op\":\"frame\",\"qpos\":[0";
    for(int i=1;i<72;i++)text+=",0";
    text+="]"+extra+"}";
    cJSON* command=cJSON_Parse(text.c_str());
    try{side=frame_request(command).follow_side;cJSON_Delete(command);return true;}
    catch(const std::exception&){cJSON_Delete(command);return false;}
}

int main(int argc,char** argv){
    try {
        if(argc!=4)throw std::runtime_error("usage: validate_follow_camera presentation.xml saved-qpos.f32 OUTPUT_DIR");
        float q[72];std::ifstream f(argv[2],std::ios::binary);f.read(reinterpret_cast<char*>(q),sizeof(q));
        if(f.gcount()!=sizeof(q)||f.peek()!=EOF)throw std::runtime_error("expected exactly72 float32 values");
        const std::string out=argv[3];const double deg=180/mjPI;int side=9;

        check(parse_side("",side)&&side==-1,"absent followSide keeps the overview");
        check(parse_side(",\"followSide\":0",side)&&side==0,"followSide 0 parsed");
        check(parse_side(",\"followSide\":1",side)&&side==1,"followSide 1 parsed");
        check(!parse_side(",\"followSide\":2",side),"followSide 2 rejected");
        check(!parse_side(",\"followSide\":0.5",side),"followSide 0.5 rejected");
        check(!parse_side(",\"followSide\":\"1\"",side),"followSide string rejected");

        float h[72];std::copy(q,q+72,h);
        FollowHeading first;yaw(h,0,10);
        check(std::abs(first.update(h,{0,true,true,0,0})*deg-10)<1e-4,"first frame snaps to the fighter heading");
        yaw(h,0,25);
        check(std::abs(first.update(h,{0,true,true,1,0})*deg-10-15*(1-std::exp(-kControlSeconds/kFollowRecentreSeconds)))<1e-4,
            "inside the dead band the view recentres over tick time");
        FollowHeading turn;yaw(h,0,0);turn.update(h,{0,true,true,0,0});yaw(h,0,90);
        check(std::abs(turn.update(h,{0,true,true,0,0})*deg-(90-kFollowDeadbandDeg))<1e-4,"a turn trails at the dead-band edge");
        FollowHeading wrap;yaw(h,0,170);wrap.update(h,{0,true,true,0,0});yaw(h,0,-170);
        const double wrapped=wrap.update(h,{0,true,true,50,0})*deg;
        check(wrapped>170||wrapped<-170,"a turn across +/-180 takes the short way");
        FollowHeading reset;yaw(h,0,0);reset.update(h,{0,true,true,100,0});yaw(h,0,120);
        check(std::abs(reset.update(h,{0,true,true,0,1})*deg-120)<1e-4,"a new generation snaps");
        FollowHeading down;yaw(h,0,30);down.update(h,{0,true,true,0,0});
        const double pitch=-mjPI/4;h[3]=float(std::cos(pitch));h[4]=0;h[5]=float(std::sin(pitch));h[6]=0;
        check(std::abs(down.update(h,{0,true,true,5,0})*deg-30)<1e-4,"a downed fighter holds the last heading");
        FollowHeading cold;const double toward=cold.update(h,{0,false,false,0,0});
        check(std::abs(wrap_angle(toward-std::atan2(h[37]-h[1],h[36]-h[0])))<1e-6,"a downed first frame faces the opponent");

        Renderer renderer(argv[1]);
        const std::string overview=renderer.frame(q);save(overview,out+"/overview.png");
        save(renderer.frame(q,{0,true,true,0,0}),out+"/follow-side0.png");
        save(renderer.frame(q,{1,true,true,0,1}),out+"/follow-side1.png");
        float c[72];std::copy(q,q+72,c);c[0]=0;c[1]=0;yaw(c,0,0);c[36]=1.0f;c[37]=.3f;yaw(c,1,-160);
        save(renderer.frame(c,{0,true,true,0,2}),out+"/follow-centre.png");
        std::copy(q,q+72,c);c[0]=-1.6f;c[1]=-.4f;yaw(c,0,15);c[36]=-.7f;c[37]=-.2f;yaw(c,1,-165);
        save(renderer.frame(c,{0,true,true,0,3}),out+"/follow-near-wall.png");
        check(renderer.frame(q)==overview,"follow frames leave the overview unchanged");
        return failures?1:0;
    }catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 2;}
}
