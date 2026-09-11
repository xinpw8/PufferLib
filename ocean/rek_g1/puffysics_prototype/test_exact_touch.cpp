#define main geometry_test_main
#include "test_cylinder_geometry.cpp"
#undef main
int main() {
    B3Shape cylinder = shape(B3_CYLINDER, .5f, b3_v(0,1,0));
    B3Shape box = shape(B3_BOX, 0, b3_v(.5f,.5f,.5f));
    contact("exact binary side touch", cylinder, box, b3_v(1,0,0), b3_q_id(), 0);
    contact("exact binary cap touch", cylinder, box, b3_v(0,1.5f,0), b3_q_id(), 0);
    return 0;
}
