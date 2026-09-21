#include "fast_observable_balance.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <limits>

namespace ob=rek_observable_balance;
namespace fast=rek_fast_observable;
static unsigned checks=0;
static void check(bool ok,const char* name){
    ++checks;if(!ok){std::fprintf(stderr,"FAIL %s\n",name);std::exit(1);}
}
static bool near(float a,float b,float tolerance=1e-5f){return std::fabs(a-b)<=tolerance;}
static fast::Input initial(){
    fast::Input in{};in.round_key=1;in.round_duration_seconds=120;
    in.round_remaining_seconds=120;in.round_active=1;
    in.root[0][0]=-1;in.root[1][0]=1;
    for(int side=0;side<2;side++){in.root[side][2]=.8f;in.root[side][3]=1;}
    return in;
}
static void yaw(fast::Input& in,int side,float theta){
    in.root[side][3]=std::cos(theta/2);in.root[side][4]=in.root[side][5]=0;
    in.root[side][6]=std::sin(theta/2);
}
// Independently construct the equivalent shared-contract snapshot. The actual
// adapter under test is the same host/device function called by fast_runtime.
static ob::Snapshot reference(const fast::Input& in,int actor){
    ob::Snapshot out{};
    for(int side=0;side<2;side++){
        std::memcpy(out.fighter[side].root_xyz,in.root[side],3*sizeof(float));
        std::memcpy(out.fighter[side].root_wxyz,in.root[side]+3,4*sizeof(float));
        out.points[side]=in.points[side];
    }
    out.actor_slot=actor;out.round_key=in.round_key;out.sample_seconds=in.sample_seconds;
    out.round_duration_seconds=in.round_duration_seconds;
    out.round_remaining_seconds=in.round_remaining_seconds;
    out.round_active=in.round_active;out.terminal=in.terminal;
    return out;
}
static void unavailable(const float* out){
    for(int side=0;side<2;side++){
        const float* row=out+side*223;
        check(row[184]==float(side),"absolute actor slot");
        check(row[202]==0&&row[204]==0&&row[205]==0,"referee explicitly unavailable");
        for(int b:{0,86}){
            check(row[b+74]==0&&row[b+75]==0,"joint availability");
            for(int k=13;k<=70;k++)check(row[b+k]==0,"joint values unavailable");
        }
        for(int k=0;k<223;k++){
            check(std::isfinite(row[k]),"finite output");
            if(!ob::structurally_available(k))check(row[k]==0,"structural padding");
        }
    }
}
static void compare(const fast::Input& now,const fast::Input* old){
    fast::History history{};float actual[446],expected[223];
    if(old)check(fast::project(history,*old,actual)==ob::kOk,"seed previous observation");
    check(fast::project(history,now,actual)==ob::kOk,"adapter status");
    for(int side=0;side<2;side++){
        auto current=reference(now,side);auto previous=old?reference(*old,side):ob::Snapshot{};
        check(ob::project(current,old?&previous:nullptr,expected)==ob::kOk,"shared reference status");
        for(int k=0;k<223;k++)check(actual[side*223+k]==expected[k],"exact shared projection mapping");
    }
    unavailable(actual);
}
int main(){
    fast::History history{};auto in=initial();float out[446];
    check(fast::project(history,in,out)==ob::kOk,"first observation");
    check(out[203]==0&&out[223+203]==0,"first history unavailable");
    check(out[86]==2&&out[87]==0&&out[223+86]==2&&near(std::fabs(out[223+87]),1),"perspective geometry");
    check(out[71]==1&&out[72]==0&&out[73]==.8f,"observed root availability and pose");
    unavailable(out);

    in.sample_seconds=.02;in.round_remaining_seconds=119.98f;
    in.root[0][0]+=.1f;in.root[0][2]+=.02f;in.points[0]=2;in.points[1]=1;
    check(fast::project(history,in,out)==ob::kOk,"second observation");
    check(out[203]==1&&out[223+203]==1,"both perspectives have history");
    check(near(out[7],5)&&near(out[9],1),"root XY and vertical differences");
    check(out[217]==2&&out[218]==1&&out[223+217]==1&&out[223+218]==2,"point deltas are perspective ordered");
    check(out[190]==2&&out[191]==1&&out[223+190]==1&&out[223+191]==2,"point totals");

    in.sample_seconds=.04;in.root[0][0]+=1;
    check(fast::project(history,in,out)==ob::kOk&&out[203]==1&&near(out[7],50),"same-round body displacement preserves history");
    in.sample_seconds=.06;in.root[0][3]=std::cos(.2f);in.root[0][4]=std::sin(.2f);
    check(fast::project(history,in,out)==ob::kOk&&near(out[72],.4f/float(ob::kPi)),"clip tilt is pose, not down flag");
    in.sample_seconds=.08;in.terminal=1;in.round_active=0;in.round_remaining_seconds=0;
    check(fast::project(history,in,out)==ob::kOk&&out[185]==4&&out[203]==1,"terminal retains last observation history");

    in=initial();in.round_key=2;
    check(fast::project(history,in,out)==ob::kOk&&out[203]==0&&out[217]==0&&out[190]==0,"new round permits point/time reset and masks history");
    in.sample_seconds=.02;
    check(fast::project(history,in,out)==ob::kOk&&out[203]==1,"new round second observation");
    history={};
    check(fast::project(history,in,out)==ob::kOk&&out[203]==0,"explicit stream reset masks history");
    in.sample_seconds=.5;
    check(fast::project(history,in,out)==ob::kOk&&out[203]==0,"history gap unavailable");
    in.sample_seconds=.52;
    check(fast::project(history,in,out)==ob::kOk&&out[203]==1,"history resumes after gap");
    check(fast::project(history,in,out)==ob::kOk&&out[203]==0,"duplicate observation time masks derivatives");
    in.sample_seconds=.51;
    check(fast::project(history,in,out)==ob::kOk&&out[203]==0,"backward time masks derivatives");

    in.sample_seconds=.53;in.points[0]=4;
    check(fast::project(history,in,out)==ob::kOk,"point increase");
    in.sample_seconds=.55;in.points[0]=3;
    check(fast::project(history,in,out)==ob::kCounterRegressed&&!history.available,"point regression fails closed");
    for(float value:out)check(value==0,"failure clears both projected rows");
    check(fast::project(history,in,out)==ob::kOk&&out[203]==0,"recovery has no invalid history");
    in.root[1][3]=0;
    check(fast::project(history,in,out)==ob::kInvalidCurrent&&!history.available,"invalid quaternion rejected");
    for(float value:out)check(value==0,"invalid pose clears output");
    in=initial();in.root[0][0]=std::numeric_limits<float>::quiet_NaN();
    check(fast::project(history,in,out)==ob::kInvalidCurrent,"nonfinite root rejected");
    in=initial();check(fast::project(history,in,nullptr)==ob::kInvalidCurrent&&!history.available,"null output rejected");
    in.round_active=2;
    check(fast::project(history,in,out)==ob::kInvalidCurrent,"invalid lifecycle flag rejected");
    in=initial();in.root[0][3]=in.root[0][5]=std::sqrt(.5f);
    check(fast::project(history,in,out)==ob::kOk&&out[71]==0&&near(out[72],.5f),"vertical forward has unavailable heading");
    unavailable(out);

    for(int i=0;i<2048;i++){
        auto old=initial(),now=initial();
        old.round_key=now.round_key=uint64_t(i/13+1);
        old.sample_seconds=.02*i;now.sample_seconds=old.sample_seconds+.02;
        old.points[0]=i%11;old.points[1]=i%7;
        now.points[0]=old.points[0]+i%3;now.points[1]=old.points[1]+i%2;
        for(int side=0;side<2;side++){
            const float phase=.013f*i+.7f*side;
            old.root[side][0]=std::sin(phase);old.root[side][1]=std::cos(phase);
            now.root[side][0]=old.root[side][0]+.02f*std::sin(phase*3);
            now.root[side][1]=old.root[side][1]+.02f*std::cos(phase*2);
            old.root[side][2]=.7f+.03f*std::cos(phase);
            now.root[side][2]=.7f+.03f*std::cos(phase+.1f);
            yaw(old,side,phase);yaw(now,side,phase+.035f);
            if(i%5==0)for(int k=3;k<7;k++)now.root[side][k]=-now.root[side][k];
        }
        if(i%7==0)now.round_key++;
        if(i%11==0)now.sample_seconds+=.5;
        compare(now,&old);if(i<32)compare(now,nullptr);
    }
    std::printf("PASS fast_observable_balance CPU checks=%u shared_projection_exact=true no_physics_no_GPU\n",checks);
}
