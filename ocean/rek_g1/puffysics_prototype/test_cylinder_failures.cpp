// SPDX-License-Identifier: MIT
// Deliberately exhaust one narrowphase/contact limit and check sticky reporting.
#include <cstdio>
#include "engine/puffysics.cuh"
int main() {
    B3World* w = new B3World; b3_world_init(w);
    B3BodyDef bd = b3_default_body(); bd.type = B3_DYNAMIC;
    int a = b3_create_body(w,&bd); bd.position = b3_v(.7f,0,0);
    int b = b3_create_body(w,&bd);
    B3ShapeDef sd = b3_default_shape();
    b3_create_cylinder_local(w,a,1,.5f,b3_v(0,0,0),b3_q_id(),&sd);
    b3_create_box(w,b,b3_v(.3f,.4f,.2f),&sd);
#if B3_MAX_CONTACTS == 1
    bd.position = b3_v(-.7f,0,0);
    int c = b3_create_body(w,&bd);
    b3_create_box(w,c,b3_v(.3f,.4f,.2f),&sd);
    unsigned expected = B3_COLLISION_CONTACT_CAPACITY;
#elif B3_CONVEX_GJK_ITERATIONS == 1
    unsigned expected = B3_COLLISION_GJK_LIMIT;
#elif B3_CONVEX_EPA_ITERATIONS == 1
    unsigned expected = B3_COLLISION_EPA_LIMIT;
#elif B3_CONVEX_VERTICES == 4
    unsigned expected = B3_COLLISION_EPA_CAPACITY;
#else
#error Compile this test with one explicit small diagnostic limit.
#endif
    b3_find_contacts(w);
    unsigned status = w->collision_status;
    bool reported = (status & expected) != 0;
    w->bodies[b].position = b3_v(10,0,0);
    b3_find_contacts(w);
    bool sticky = (w->collision_status & expected) != 0;
    std::printf("expected=%u status=%u sticky=%u failed_queries=%u overflow=%u pass=%d\n",
        expected,status,w->collision_status,w->collision_failed_queries,
        w->collision_contact_overflows,int(reported && sticky));
    delete w;
    return reported && sticky ? 0 : 1;
}
