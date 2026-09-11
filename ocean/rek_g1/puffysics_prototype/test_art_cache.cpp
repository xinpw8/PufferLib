// SPDX-License-Identifier: MIT
// CPU-only deterministic randomized operator and contact-phase comparison.
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#define B3_MAX_JOINTS 64
#define B3_ART_CONTACTS 0
static float test_armature[64];
#define B3_ART_JOINT_ARMATURE(j) ((j) >= 0 ? test_armature[(j)] : 0.0f)
#define B3_ART_CACHE_DIAGNOSTICS 1
#include "engine/puffysics.cuh"
#if TEST_ORIGINAL
#include "engine/b3_art.cuh"
#else
#include "engine/b3_art_cached.cuh"
#endif

static unsigned state;
static unsigned random_u() { state ^= state << 13; state ^= state >> 17; state ^= state << 5; return state; }
static float random_f() { return float(random_u() >> 8) * (1.0f / 16777216.0f); }
static B3Vec3 random_v(float magnitude) {
    return b3_v((2*random_f()-1)*magnitude, (2*random_f()-1)*magnitude, (2*random_f()-1)*magnitude);
}
static void require(bool ok, const char* reason) {
    if (!ok) { std::fprintf(stderr,"FAIL %s\n",reason); std::exit(1); }
}
static void put(std::vector<float>& out, B3Vec3 p) { out.push_back(p.x); out.push_back(p.y); out.push_back(p.z); }
static void velocities(std::vector<float>& out, const B3World* w) {
    for (int i = 0; i < w->body_count; i++) { put(out,w->bodies[i].lin_vel); put(out,w->bodies[i].ang_vel); }
}
static void body_state(std::vector<float>& out, const B3World* w) {
    for (int i = 0; i < w->body_count; i++) {
        const B3Body* b=&w->bodies[i]; put(out,b->position); put(out,b->center);
        put(out,b->rotation.v); out.push_back(b->rotation.s);
        put(out,b->delta_pos); put(out,b->delta_rot.v); out.push_back(b->delta_rot.s);
    }
    velocities(out,w);
}
static void contacts(std::vector<float>& out, const B3World* w) {
    for (int i=0;i<w->contact_count;i++) {
        const B3Contact* c=&w->contacts[i];
        for (int j=0;j<c->point_count;j++) {
            out.push_back(c->points[j].normal_impulse); out.push_back(c->points[j].normal_mass);
            out.push_back(c->points[j].total_normal);
        }
        out.push_back(c->friction_impulse.x); out.push_back(c->friction_impulse.y);
        out.push_back(c->twist_impulse); put(out,c->rolling_impulse);
    }
}
static B3World* make_world(unsigned seed) {
    state=seed;
    B3World* w=new B3World; b3_world_init(w);
    B3BodyDef def=b3_default_body(); b3_create_body(w,&def);
    for (int tree=0;tree<3;tree++) {
        int parent=0;
        for (int link=0;link<7;link++) {
            def=b3_default_body(); def.type=B3_DYNAMIC;
            def.position=b3_add(b3_v(float(tree),float(link)*.2f,0),random_v(.07f));
            def.rotation=b3_q_axis_angle(b3_norm(random_v(1)),random_f());
            def.lin_vel=random_v(.5f); def.ang_vel=random_v(.3f);
            int child=b3_create_body(w,&def);
            b3_set_inertial(w,child,1+4*random_f(),random_v(.02f),
                b3_v(.02f+.1f*random_f(),.02f+.1f*random_f(),.02f+.1f*random_f()));
            B3Body* b=&w->bodies[child]; b->delta_pos=random_v(.003f);
            b->delta_rot=b3_q_axis_angle(b3_norm(random_v(1)),random_f()*.02f);
            b->force=random_v(3); b->torque=random_v(.2f);
            // Two floating roots and one fixed-root tree.
            if (link || tree==2) {
                B3Vec3 anchor=b3_mul(b3_add(w->bodies[parent].position,b->position),.5f);
                B3Vec3 la=b3_inv_rotate(w->bodies[parent].rotation,b3_sub(anchor,w->bodies[parent].position));
                B3Vec3 lb=b3_inv_rotate(b->rotation,b3_sub(anchor,b->position));
                int joint=b3_create_revolute(w,parent,child,la,lb,b3_norm(random_v(1)));
                test_armature[joint]=.001f+.02f*random_f();
            }
            parent=child;
        }
    }
    return w;
}
static unsigned factor_builds=0,rhs_calls=0;
static void collect_counters(const B3Art* art) {
#if !TEST_ORIGINAL
    factor_builds+=art->linear_factor_builds; rhs_calls+=art->linear_rhs_calls;
#else
    (void)art;
#endif
}
static void operator_test(unsigned seed,std::vector<float>& out) {
    B3World* w=make_world(seed); B3Art* art=new B3Art;
    require(b3_art_bind(art,w),"bind operator");
    for (int phase=0;phase<2;phase++) {
#if !TEST_ORIGINAL && B3_ART_FIXED_POSE_CACHE
        b3_art_linear_cache_begin(art,w);
#endif
        for (int query=0;query<48;query++) {
            B3ArtRow rows[4]; float x[4],y[4];
            for(int r=0;r<4;r++) {
                int a=int(random_u()%unsigned(w->body_count));
                int b=int(random_u()%unsigned(w->body_count-1)); if(b>=a)b++;
                require(b3_art_make_row(art,a,b,random_v(.15f),random_v(.15f),b3_norm(random_v(1)),&rows[r]),"row");
                rows[r].torque=(query+r)%3==0; x[r]=2*random_f()-1;
            }
            b3_art_delassus_apply(art,w,rows,x,y,4);
            for(float v:y)out.push_back(v);
            for(int link=0;link<art->n_links;link++){ put(out,art->a[link].w); put(out,art->a[link].v); }
            b3_art_apply_impulse(art,w,&rows[0],.01f*(2*random_f()-1));
            velocities(out,w);
        }
#if !TEST_ORIGINAL && B3_ART_FIXED_POSE_CACHE
        b3_art_linear_cache_end(art);
        require(!art->linear_factor_scope,"explicit end");
#endif
        // Exercise invalidation by the real position integrator between scopes.
        b3_art_integrate_pos(art,w,.002f,500.0f);
#if !TEST_ORIGINAL
        require(!art->linear_factor_scope,"position invalidates");
#endif
        body_state(out,w);
    }
    collect_counters(art); delete art; delete w;
}
static void contact_test(unsigned seed,std::vector<float>& out) {
    B3World* w=make_world(seed); B3Art* art=new B3Art;
    require(b3_art_bind(art,w),"bind contacts");
    for(int i=0;i<12;i++) {
        int a=i%3 ? 1+int(random_u()%7) : 0;
        int b=8+int(random_u()%14);
        B3Contact* c=&w->contacts[w->contact_count++]; *c={};
        c->body_a=a;c->body_b=b;c->point_count=1+i%2;
        c->normal=b3_norm(random_v(1)); c->tangent1=b3_perp(c->normal);
        c->tangent2=b3_cross(c->tangent1,c->normal);
        c->friction=.3f+.4f*random_f();c->rolling=.001f;
        for(int j=0;j<c->point_count;j++) {
            B3Point* p=&c->points[j];p->r_a=random_v(.04f);p->r_b=random_v(.04f);
            p->base_sep=-.004f-.004f*random_f();p->normal_impulse=.01f*random_f();
        }
        b3_prepare_one_contact(c,&w->bodies[a],&w->bodies[b],
            b3_make_soft(30,10,.002f),b3_make_soft(60,5,.002f));
    }
    b3_art_solve_contacts(art,w,500,w->contact_speed,1,4);
#if !TEST_ORIGINAL
    require(!art->linear_factor_scope,"biased phase end");
#endif
    body_state(out,w);contacts(out,w);
    b3_art_integrate_pos(art,w,.002f,500);
    b3_art_solve_contacts(art,w,500,w->contact_speed,0,4);
#if !TEST_ORIGINAL
    require(!art->linear_factor_scope,"unbiased phase end");
#endif
    body_state(out,w);contacts(out,w);
    // Noncached full ABA must invalidate factors and preserve force behavior.
    b3_art_integrate_vel(art,w,.002f);body_state(out,w);
    collect_counters(art); delete art;delete w;
}
int main(int argc,char** argv) {
    require(argc==2,"output path");std::vector<float> out;
    for(unsigned seed=1;seed<=12;seed++) {
        operator_test(132947u*seed+19,out);contact_test(927491u*seed+77,out);
    }
    for(float v:out)require(std::isfinite(v),"finite outputs");
    FILE* f=std::fopen(argv[1],"wb");require(f!=nullptr,"open output");
    require(std::fwrite(out.data(),sizeof(float),out.size(),f)==out.size(),"write output");
    require(std::fclose(f)==0,"close output");
    std::printf("{\"original\":%d,\"cache\":%d,\"seeds\":12,\"floats\":%zu,\"factor_builds\":%u,\"rhs_calls\":%u}\n",
        TEST_ORIGINAL,B3_ART_FIXED_POSE_CACHE,out.size(),factor_builds,rhs_calls);
}
