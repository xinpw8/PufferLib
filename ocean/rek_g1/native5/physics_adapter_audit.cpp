// Read-only adapter arithmetic audit. No CUDA initialization or physics steps.
#include "physics.cuh"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <memory>
#include <vector>
#define B3_ART_CONTACTS 0
#define B3_MAX_JOINTS 64
#include "../puffysics_prototype/engine/puffysics.cuh"

static B3Quat quaternion(const float* p) {return b3_q(p[0],p[1],p[2],p[3]);}
int main(int argc,char** argv) {
    if(argc!=3){std::fprintf(stderr,"usage: physics_adapter_audit XML EXPORT\n");return 1;}
    try {
        std::unique_ptr<rek5::Physics,decltype(&rek5::physics_close)> p(rek5::physics_load_model(argv[1],argv[2]),rek5::physics_close);
        int crossings=0,unstable_proxy=0,worst=-1;float max_proxy=0;
        std::printf("{\"cpu_physics_steps\":0,\"gpu_initializations\":0,\"wrapped_limit_mismatches\":[");
        for(int j=0;j<58;j++) {
            const float* joint=p->packed_joints.data()+j*22;
            const float lo=joint[14],hi=joint[15];
            if(hi>B3_PI || lo<-B3_PI) {
                const float legal=hi>B3_PI?.5f*(B3_PI+hi):.5f*(lo-B3_PI);
                const float wrapped=b3_twist(b3_q_axis_angle(b3_v(0,0,1),legal));
                const float midpoint=.5f*(lo+hi);
                const float gathered=wrapped+2*B3_PI*nearbyintf((midpoint-wrapped)/(2*B3_PI));
                if(!(legal>=lo&&legal<=hi) || !(wrapped<lo||wrapped>hi))return 2;
                std::printf("%s{\"joint\":%d,\"legal_relative_angle\":%.9g,\"solver_wrapped_angle\":%.9g,\"controller_unwrapped_angle\":%.9g,\"lower\":%.9g,\"upper\":%.9g}",
                        crossings?",":"",j,legal,wrapped,gathered,lo,hi);crossings++;
            }
            const int parent=int(joint[0]),child=int(joint[1]);
            const float* parent_body=p->packed_bodies.data()+parent*11;
            const B3Vec3 axis=b3_rotate(quaternion(parent_body+3),b3_v(joint[10],joint[11],joint[12]));
            float response=0;
            for(int body:{parent,child}) {
                const float* packed=p->packed_bodies.data()+body*11;
                const B3Vec3 local=b3_rotate(b3_qconj(quaternion(packed+3)),axis);
                response+=local.x*local.x/packed[8]+local.y*local.y/packed[9]+local.z*local.z/packed[10];
            }
            // Explicit damping's scalar relative-rotation update multiplier is
            // 1-h*c*(axis dot (I_parent^-1+I_child^-1) axis). This is a local
            // unconstrained proxy, not an eigenanalysis of the full joint tree.
            const float proxy=.002f*joint[20]*response;
            if(proxy>2)unstable_proxy++;
            if(proxy>max_proxy){max_proxy=proxy;worst=j;}
        }
        std::printf("],\"crossing_hinges\":%d,\"passive_damping_local_proxy\":{\"hinges_above_scalar_stability_limit\":%d,\"worst_joint\":%d,\"max_h_c_inverse_inertia\":%.9g,\"scalar_limit\":2},\"diagnostic_only\":true}\n",
                crossings,unstable_proxy,worst,max_proxy);
        return 0;
    }catch(const std::exception& error){std::fprintf(stderr,"%s\n",error.what());return 1;}
}
