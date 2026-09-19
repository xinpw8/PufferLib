#include "round_reward.h"
#include <cmath>
#include <cstdio>
#include <stdexcept>

static unsigned checks=0;
static void check(bool ok,const char* what){++checks;if(!ok)throw std::runtime_error(what);}
static void close(double a,double b,double tolerance,const char* what){check(std::abs(a-b)<=tolerance,what);}

int main(){try{
    using namespace rek5_round_reward;
    for(int a=0;a<=50;a++)for(int b=0;b<=50;b++){
        close(potential(a,b),-potential(b,a),1e-7,"potential is not antisymmetric");
        check(std::abs(potential(a,b))<1,"unbounded potential");
        for(int da=0;da<=5;da++)for(int db=0;db<=5;db++){
            close(value(PointDifference,.999f,a,b,a+da,b+db,false,-1,0),da-db,0,"legacy point reward changed");
            close(value(PointDifference,.999f,a,b,a+da,b+db,true,0,0),da-db,0,"legacy terminal double counting");
            const float x=value(RoundOutcome,.999f,a,b,a+da,b+db,false,-1,0);
            const float y=value(RoundOutcome,.999f,b,a,b+db,a+da,false,-1,1);
            close(x,-y,1e-7,"opponent penalty differs from own award");
        }
    }
    // A received +5 countout is score feedback, not a round boundary.
    close(value(PointDifference,1,0,0,0,5,false,-1,0),-5,0,"countout penalty missing");
    close(value(RoundOutcome,1,0,0,0,5,false,-1,0),-.5,0,"countout falsely treated as terminal loss");
    close(value(RoundOutcome,1,0,5,0,5,false,-1,0),0,0,"spawn reset repeats countout reward");
    close(value(RoundOutcome,1,5,0,5,0,true,0,0),.5,0,"terminal outcome missing potential removal");
    close(value(RoundOutcome,1,3,3,3,3,true,-1,0),0,0,"draw earns win reward");
    double worst=0;
    const float gammas[]={.99f,.999f,.9998844821426083f,1.f};
    for(float gamma:gammas)for(int seed=1;seed<=64;seed++)for(int side=0;side<2;side++){
        int points[2]={seed%7,(seed/7)%7};
        const double initial=potential(points[side],points[side^1]);
        double discounted=0,weight=1;int winner=-1;
        for(int t=0;t<6000;t++){
            const int previous[2]={points[0],points[1]};
            if((t+seed*31)%307==0)points[(t/307+seed)%2]++;
            if((t+seed*17)%991==0)points[(t/991+seed)%2]+=5;
            const bool terminal=t==5999;
            if(terminal)winner=points[0]==points[1]?-1:(points[0]>points[1]?0:1);
            const float reward=value(RoundOutcome,gamma,previous[side],previous[side^1],points[side],points[side^1],terminal,winner,side);
            discounted+=weight*reward;
            if(!terminal)weight*=gamma;
        }
        const double expected=weight*(winner<0?0:(winner==side?1:-1))-initial;
        worst=std::fmax(worst,std::abs(discounted-expected));
        close(discounted,expected,2e-4,"dense rewards changed discounted terminal objective");
    }
    // For equal-length rounds starting 0:0, margin and number of exchanges
    // cannot improve an outcome-shaped loss beyond an outcome-shaped win.
    close(value(RoundOutcome,1,0,0,1,0,true,0,0),1,0,"one-point win objective");
    close(value(RoundOutcome,1,0,0,100,0,true,0,0),1,0,"score farming changes win objective");
    close(value(RoundOutcome,1,0,0,99,100,true,1,0),-1,0,"high-scoring loss reward");
    std::printf("{\"test\":\"round_reward\",\"checks\":%u,\"full_120s_trajectories\":512,\"max_telescoping_error\":%.12g,\"status\":\"passed\",\"physics_parity\":false}\n",checks,worst);
    return 0;
}catch(const std::exception& e){std::fprintf(stderr,"reward test: %s\n",e.what());return 1;}}
