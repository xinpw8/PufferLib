#include "primitive_contacts.cuh"
#include "primitive_motion.cuh"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>

namespace {
using rek5_primitive::Shape;
using rek5_primitive::Sphere;
using rek5_primitive::Capsule;
using rek5_primitive::Box;
int checks=0;

void require(bool value,const char* label) {
    ++checks;
    if(!value) { std::fprintf(stderr,"FAIL: %s\n",label); std::exit(1); }
}
Shape shape(int kind,float x,float y,float z,float a,float b=0,float c=0) {
    return {kind,{x,y,z},{1,0,0,0,1,0,0,0,1},{a,b,c}};
}
void expect(const Shape& a,const Shape& b,bool value,const char* label) {
    require(rek5_primitive::overlap(a,b)==value,label);
    require(rek5_primitive::overlap(b,a)==value,"pair symmetry");
}
// Independent host transform: Rz(yaw)*Ry(pitch)*Rx(roll), left-multiplied.
Shape transform(const Shape& s,float yaw,float pitch,float roll,
                float tx=0,float ty=0,float tz=0) {
    const float c=std::cos(yaw),d=std::sin(yaw),e=std::cos(pitch),f=std::sin(pitch);
    const float g=std::cos(roll),h=std::sin(roll);
    const float r[9]={c*e,c*f*h-d*g,c*f*g+d*h,
                      d*e,d*f*h+c*g,d*f*g-c*h,-f,e*h,e*g};
    Shape out=s;
    const float t[3]={tx,ty,tz};
    for(int i=0;i<3;i++) {
        out.center[i]=t[i];
        for(int k=0;k<3;k++) out.center[i]+=r[3*i+k]*s.center[k];
        for(int j=0;j<3;j++) {
            out.axes[3*i+j]=0;
            for(int k=0;k<3;k++) out.axes[3*i+j]+=r[3*i+k]*s.axes[3*k+j];
        }
    }
    return out;
}
void analytic() {
    const Shape unit=shape(Sphere,0,0,0,1);
    expect(unit,shape(Sphere,2,0,0,1),true,"sphere touching");
    expect(unit,shape(Sphere,2.00001f,0,0,1),false,"sphere no positive margin");
    expect(unit,shape(Sphere,0,0,0,0),true,"sphere concentric point");
    expect(shape(Sphere,10000,0,0,.125f),shape(Sphere,10000.25f,0,0,.125f),true,"large coordinates touching");
    expect(shape(Sphere,0,0,0,.000001f),shape(Sphere,.000003f,0,0,.000001f),false,"small shapes separated");

    const Shape cap=shape(Capsule,0,0,0,.5f,2);
    expect(cap,shape(Sphere,1,0,0,.5f),true,"sphere capsule side touching");
    expect(cap,shape(Sphere,0,0,3,.5f),true,"sphere capsule end touching");
    expect(cap,shape(Sphere,1.01f,0,0,.5f),false,"sphere capsule side outside");
    expect(cap,shape(Sphere,0,0,3.01f,.5f),false,"sphere capsule end outside");
    expect(shape(Capsule,0,0,0,1,0),shape(Sphere,2,0,0,1),true,"zero length capsule");

    const Shape box=shape(Box,0,0,0,1,1,1);
    expect(box,shape(Sphere,0,0,0,0),true,"point inside box");
    expect(box,shape(Sphere,1.5f,0,0,.5f),true,"sphere box face touching");
    expect(box,shape(Sphere,1.501f,0,0,.5f),false,"sphere box face outside");
    expect(box,shape(Sphere,1.3f,1.3f,0,.5f),true,"sphere box edge inside radius");
    expect(box,shape(Sphere,1.4f,1.4f,0,.5f),false,"sphere box edge outside radius");
    expect(box,shape(Sphere,1.25f,1.25f,1.25f,.5f),true,"sphere box corner inside radius");
    expect(box,shape(Sphere,1.4f,1.4f,1.4f,.5f),false,"sphere box corner outside radius");

    expect(cap,shape(Capsule,1,0,0,.5f,2),true,"parallel capsule sides touching");
    expect(cap,shape(Capsule,1.001f,0,0,.5f,2),false,"parallel capsules separated");
    expect(cap,shape(Capsule,0,0,5,.5f,2),true,"collinear capsule ends touching");
    expect(cap,shape(Capsule,0,0,5.001f,.5f,2),false,"collinear capsules separated");
    Shape horizontal=shape(Capsule,0,0,0,.125f,3);
    horizontal.axes[0]=0;horizontal.axes[2]=1;horizontal.axes[6]=-1;horizontal.axes[8]=0;
    expect(cap,horizontal,true,"crossing capsule interiors");
    horizontal.center[1]=.626f;
    expect(cap,horizontal,false,"skew capsule interiors separated");
    horizontal.center[1]=.625f;
    expect(cap,horizontal,true,"skew capsule interiors touching");
    expect(shape(Capsule,0,0,0,.25f,0),shape(Capsule,.5f,0,0,.25f,0),true,"two zero length capsules");

    expect(box,shape(Capsule,1.125f,0,0,.125f,3),true,"capsule box side touching beyond endpoints");
    expect(box,shape(Capsule,1.126f,0,0,.125f,3),false,"capsule box side separated");
    expect(box,shape(Capsule,0,0,4,.5f,2.5f),true,"capsule box end touching");
    expect(box,shape(Capsule,0,0,4.01f,.5f,2.5f),false,"capsule box end separated");
    expect(box,shape(Capsule,1.25f,1.25f,0,.4f,3),true,"capsule box long edge near");
    expect(box,shape(Capsule,1.25f,1.25f,0,.3f,3),false,"capsule box long edge far");
    expect(box,shape(Capsule,0,0,0,0,5),true,"zero radius capsule crosses box");
    expect(box,shape(Capsule,0,0,0,0,0),true,"point capsule inside box");
    Shape diagonal=transform(shape(Capsule,0,0,0,.02f,3),.4f,.8f,.3f);
    expect(box,diagonal,true,"oblique capsule crosses box interior");
    expect(box,shape(Capsule,1.25f,1.25f,2,.45f,.75f),true,"capsule box endpoint corner near");
    expect(box,shape(Capsule,1.25f,1.25f,2,.4f,.75f),false,"capsule box endpoint corner far");

    expect(box,shape(Box,2,0,0,1,1,1),true,"boxes face touching");
    expect(box,shape(Box,2.00001f,0,0,1,1,1),false,"boxes no positive margin");
    expect(box,shape(Box,2,2,0,1,1,1),true,"boxes edge touching");
    expect(box,shape(Box,2,2,2,1,1,1),true,"boxes corner touching");
    expect(box,shape(Box,0,0,0,.1f,.1f,.1f),true,"box containment");
    Shape rotated=transform(shape(Box,0,0,0,1,.1f,.1f),.7853981634f,0,0);
    rotated.center[0]=1.8f;
    expect(box,rotated,false,"rotated thin box separated");
    rotated.center[0]=1.5f;
    expect(box,rotated,true,"rotated thin box overlaps");
    Shape almost=transform(shape(Box,2.0001f,0,0,1,1,1),.000001f,0,0);
    expect(box,almost,false,"nearly parallel boxes separated");
    // Orthogonal long rods: each face normal sees the other's long axis, but
    // their long-axis cross product separates along world Z. Each Z extent is
    // sqrt(2)*.1, so .29 exceeds their combined .282843 projected thickness.
    const Shape rod_a=transform(shape(Box,0,0,0,1,.1f,.1f),0,0,.7853981634f);
    Shape rod_b=transform(shape(Box,0,0,0,1,.1f,.1f),1.5707963268f,0,.7853981634f);
    rod_b.center[2]=.29f;
    expect(rod_a,rod_b,false,"edge cross axis separates rods");
    rod_b.center[2]=.27f;
    expect(rod_a,rod_b,true,"edge cross axis rod overlap");
    expect(shape(Box,0,0,0,0,0,0),box,true,"point box contained");
    expect(shape(-1,0,0,0,1),unit,false,"invalid kind rejected");
}

void distance_cases() {
    using rek5_primitive::detail::Vec;
    using rek5_primitive::detail::segment_aabb2;
    using rek5_primitive::detail::segment_segment2;
    const float half[3]={1,1,1};
    require(std::fabs(segment_aabb2({-4,2,0},{4,2,0},half)-1)<1e-6f,"segment AABB interior minimum");
    require(std::fabs(segment_aabb2({2,2,0},{2,2,3},half)-2)<1e-6f,"segment AABB edge minimum");
    require(std::fabs(segment_aabb2({2,2,2},{3,3,3},half)-3)<1e-6f,"segment AABB endpoint minimum");
    require(segment_aabb2({-4,0,0},{4,0,0},half)==0,"segment AABB slab crossing");
    require(segment_aabb2({0,0,0},{0,0,0},half)==0,"segment AABB degenerate inside");
    require(std::fabs(segment_segment2({-1000,0,0},{1000,0,0},{-1000,-.01f,0},{1000,.01f,0}))<1e-12f,
            "nearly parallel long segments cross at interiors");
    require(std::fabs(segment_segment2({-1000,0,0},{1000,0,0},{-1000,-.01f,.5f},{1000,.01f,.5f})-.25f)<1e-6f,
            "nearly parallel long skew segments");
}

void independent_distance_oracle() {
    // A host-only double-precision convex minimizer is independent of the
    // production slab-breakpoint quadratic implementation.
    std::mt19937 rng(91321);
    std::uniform_real_distribution<float> position(-8,8),size(.01f,2);
    for(int trial=0;trial<3000;trial++) {
        const float p[3]={position(rng),position(rng),position(rng)};
        const float q[3]={position(rng),position(rng),position(rng)};
        const float half[3]={size(rng),size(rng),size(rng)};
        auto distance=[&](double t) {
            double total=0;
            for(int k=0;k<3;k++) {
                const double x=double(p[k])+t*(double(q[k])-p[k]);
                const double excess=std::fmax(std::fabs(x)-half[k],0.0);
                total+=excess*excess;
            }
            return total;
        };
        double lo=0,hi=1;
        for(int iteration=0;iteration<100;iteration++) {
            const double a=lo+(hi-lo)/3,b=hi-(hi-lo)/3;
            if(distance(a)<distance(b)) hi=b;else lo=a;
        }
        const double expected=std::fmin(std::fmin(distance(0),distance(1)),distance((lo+hi)*.5));
        const float actual=rek5_primitive::detail::segment_aabb2({p[0],p[1],p[2]},{q[0],q[1],q[2]},half);
        require(std::fabs(actual-expected)<2e-5*(1+expected),"segment AABB independent double convex oracle");
    }
}

void invariance() {
    std::mt19937 rng(7391);
    std::uniform_real_distribution<float> position(-2.5f,2.5f),angle(-3,3),size(.03f,.7f);
    for(int i=0;i<12000;i++) {
        const int kind_a=i%3,kind_b=(i/3)%3;
        Shape a=transform(shape(kind_a,position(rng),position(rng),position(rng),size(rng),size(rng),size(rng)),angle(rng),angle(rng),angle(rng));
        Shape b=transform(shape(kind_b,position(rng),position(rng),position(rng),size(rng),size(rng),size(rng)),angle(rng),angle(rng),angle(rng));
        const bool baseline=rek5_primitive::overlap(a,b);
        require(rek5_primitive::overlap(b,a)==baseline,"random pair symmetry");
        const float yaw=angle(rng),pitch=angle(rng),roll=angle(rng);
        const Shape ar=transform(a,yaw,pitch,roll),br=transform(b,yaw,pitch,roll);
        require(rek5_primitive::overlap(ar,br)==baseline,"random common rotation invariance");
        const Shape at=transform(a,0,0,0,3.25f,-4.5f,1.75f),bt=transform(b,0,0,0,3.25f,-4.5f,1.75f);
        require(rek5_primitive::overlap(at,bt)==baseline,"random common translation invariance");
    }
}

void matrix_roundtrip(const Shape& input) {
    float quaternion[4],matrix[9];
    rek5_primitive::matrix_quaternion(input.axes,quaternion);
    float length2=0;
    for(float value:quaternion) {
        require(std::isfinite(value),"matrix quaternion finite");
        length2+=value*value;
    }
    require(std::fabs(length2-1)<1e-6f,"matrix quaternion unit length");
    rek5_primitive::quaternion_matrix(quaternion,matrix);
    for(int i=0;i<9;i++) require(std::fabs(matrix[i]-input.axes[i])<1e-6f,"matrix quaternion roundtrip");
}

void motion() {
    const Shape identity=shape(Box,0,0,0,.3f,.4f,.5f);
    matrix_roundtrip(identity);
    Shape half_turn_x=identity,half_turn_y=identity,half_turn_z=identity;
    half_turn_x.axes[4]=half_turn_x.axes[8]=-1;
    half_turn_y.axes[0]=half_turn_y.axes[8]=-1;
    half_turn_z.axes[0]=half_turn_z.axes[4]=-1;
    matrix_roundtrip(half_turn_x);matrix_roundtrip(half_turn_y);matrix_roundtrip(half_turn_z);
    std::mt19937 rng(11831);
    std::uniform_real_distribution<float> angle(-3.14159265f,3.14159265f);
    for(int i=0;i<1000;i++) {
        const Shape local=transform(shape(Box,.2f,.3f,.4f,.3f,.4f,.5f),angle(rng),angle(rng),angle(rng));
        matrix_roundtrip(local);
        const float yaw=angle(rng);
        const Shape actual=rek5_primitive::world_shape(local,2,-1,yaw);
        const Shape expected=transform(local,yaw,0,0,2,-1,0);
        for(int k=0;k<3;k++) require(std::fabs(actual.center[k]-expected.center[k])<1e-6f,"world yaw center");
        for(int k=0;k<9;k++) require(std::fabs(actual.axes[k]-expected.axes[k])<1e-6f,"world yaw axes");
        const Shape middle=rek5_primitive::interpolate_shape(identity,local,.375f);
        for(int k=0;k<3;k++) {
            require(std::fabs(middle.center[k]-.375f*local.center[k])<1e-6f,"interpolated center");
            for(int j=0;j<3;j++) {
                float product=0;
                for(int r=0;r<3;r++) product+=middle.axes[3*r+k]*middle.axes[3*r+j];
                require(std::fabs(product-(k==j?1.f:0.f))<1e-6f,"interpolated orthonormal axes");
            }
        }
        const auto& m=middle.axes;
        const float determinant=m[0]*(m[4]*m[8]-m[5]*m[7])-m[1]*(m[3]*m[8]-m[5]*m[6])+m[2]*(m[3]*m[7]-m[4]*m[6]);
        require(std::fabs(determinant-1)<1e-6f,"interpolated proper rotation");
    }
    const Shape endpoint0=rek5_primitive::interpolate_shape(identity,half_turn_z,0);
    const Shape endpoint1=rek5_primitive::interpolate_shape(identity,half_turn_z,1);
    for(int i=0;i<9;i++) {
        require(endpoint0.axes[i]==identity.axes[i],"interpolation exact old endpoint");
        require(endpoint1.axes[i]==half_turn_z.axes[i],"interpolation exact new endpoint");
    }
    const Shape target=shape(Sphere,0,0,0,.125f);
    const Shape start=shape(Sphere,-1,0,0,.125f);
    const Shape end=shape(Sphere,1,0,0,.125f);
    require(!rek5_primitive::sampled_overlap(start,end,target,target,1),"one sample misses middle crossing");
    require(rek5_primitive::sampled_overlap(start,end,target,target,4),"four samples see middle crossing");
    const Shape touching_end=shape(Sphere,.25f,0,0,.125f);
    require(rek5_primitive::sampled_overlap(start,touching_end,target,target,1),"one sample includes touching endpoint");
    require(rek5_primitive::sampled_overlap(start,touching_end,target,target,4),"four samples include touching endpoint");
    require(!rek5_primitive::sampled_overlap(target,shape(Sphere,2,0,0,.125f),target,target,4),"old endpoint is excluded");
    // This small target lies between t=.25 and t=.5: the deliberate miss makes
    // explicit that rigid sampling is not continuous collision detection.
    const Shape tiny_start=shape(Sphere,-.75f,0,0,.01f),tiny_end=shape(Sphere,1.25f,0,0,.01f);
    const Shape tiny_target=shape(Sphere,0,0,0,.01f);
    require(!rek5_primitive::sampled_overlap(tiny_start,tiny_end,tiny_target,tiny_target,4),"four samples retain a between-sample miss");
    require(rek5_primitive::sampled_overlap(tiny_start,tiny_end,tiny_target,tiny_target,8),"eight samples see that specific crossing");
}
} // namespace

int main() {
    analytic();distance_cases();independent_distance_oracle();invariance();motion();
    std::printf("{\"event\":\"primitive_contacts_tests\",\"checks\":%d,\"passed\":true,\"contact_margin_m\":0,\"static_geometry_only\":true,\"dynamic_parity_claim\":false}\n",checks);
}
