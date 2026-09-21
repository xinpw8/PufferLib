#include "physical_bot1_geometry.h"
#include <cstdio>
#include <cstdlib>
#include <limits>

namespace {
int checks = 0;
void require(bool value) { ++checks; if (!value) std::abort(); }
void sample(const float* own, float x, float y, float z, float distance, float angle) {
    const float other[7] = {x,y,z,1,0,0,0};
    float actual_distance = -1, actual_angle = -999;
    require(rek5_bot1_physical::geometry(own, other, actual_distance, actual_angle));
    require(std::fabs(actual_distance - distance) < 0.0001f);
    require(std::fabs(actual_angle - angle) < 0.001f);
}
}
int main() {
    float own[7] = {0,0,1,1,0,0,0};
    sample(own,1,0,1,1,0);
    sample(own,-1,0,1,1,180);
    sample(own,-1,-0.0f,1,1,180);
    sample(own,0,1,1,1,-90);
    sample(own,0,-1,1,1,90);
    sample(own,3,4,100,5,-53.1301024f);
    sample(own,3,4,-100,5,-53.1301024f);
    sample(own,0,0,100,0,0);
    sample(own,-0.03f,0,1,0.03f,0);
    sample(own,-0.04f,0,1,0.04f,180);
    own[3]=0; own[6]=1;
    sample(own,1,0,1,1,180);
    sample(own,0,1,1,1,90);
    own[3]=0.7071067812f; own[6]=0.7071067812f;
    sample(own,0,1,1,1,0);
    sample(own,1,0,1,1,90);
    own[3]=0.7071067812f; own[5]=0.7071067812f; own[6]=0;
    sample(own,-1,0,1,1,0); // Near-vertical forward uses the native guard.
    own[3]=2; own[5]=0;
    sample(own,0,-1,1,1,90); // Normalized physical quaternion projection.
    float other[7]={1,0,1,1,0,0,0}, d=7, angle=8;
    own[3]=0;
    require(!rek5_bot1_physical::geometry(own,other,d,angle));
    require(d==7 && angle==8);
    own[3]=1; own[0]=std::numeric_limits<float>::quiet_NaN();
    require(!rek5_bot1_physical::geometry(own,other,d,angle));
    own[0]=0; other[1]=std::numeric_limits<float>::infinity();
    require(!rek5_bot1_physical::geometry(own,other,d,angle));
    std::printf("{\"test\":\"physical_bot1_geometry\",\"checks\":%d,\"passed\":true,\"authentic_parity\":false}\n",checks);
}
