// Reference trace from the recovered C++ header for the JS port parity test.
// Build: g++ -std=c++17 -O1 -I.. bot1_reference.cpp -o bot1_reference
#include "../../native5/native_bot1.cuh"
#include <cstdio>
#include <cstdint>
using namespace rek5_bot1;
static uint64_t lcg = 0x243f6a8885a308d3ull;
static uint32_t rnd() { lcg = lcg * 6364136223846793005ull + 1442695040888963407ull; return uint32_t(lcg >> 33); }
static float uni(float lo, float hi) { return lo + (hi - lo) * float(rnd() % 1000000) / 1000000.f; }
int main(int argc, char** argv) {
  int steps = argc > 1 ? atoi(argv[1]) : 20000;
  Catalog cat{}; const int limbs[17] = {1,1,1,2,2,1,3,3,4,4,1,2,1,1,1,1,1};
  for (int k = 0; k < 17; k++) cat.primary_limb[k] = limbs[k];
  State s{}; activate(s, 12345u);
  float dist = 1.8f, ang = 0.f, t = 0.f; int punchLeft = 0; bool down = false; int downLeft = 0;
  for (int n = 0; n < steps; n++) {
    float dt = (rnd() % 10 == 0) ? uni(0.005f, 0.04f) : 0.02f;
    dist += uni(-0.03f, 0.028f); if (dist < 0.2f) dist = 0.2f; if (dist > 2.5f) dist = 2.5f;
    ang += uni(-6.f, 6.f); if (rnd() % 400 == 0) ang = uni(-180.f, 180.f);
    if (ang > 180.f) ang -= 360.f; if (ang < -180.f) ang += 360.f;
    if (downLeft > 0) { downLeft--; if (!downLeft) down = false; } else if (rnd() % 900 == 0) { down = true; downLeft = 60 + rnd() % 120; }
    t += dt;
    Input in{dist, ang, dt, t, t, punchLeft > 0, down, false, (n % 3000) > 20};
    struct R { uint32_t& state; uint32_t next(){uint32_t x=state;x^=x<<13;x^=x>>17;x^=x<<5;return state=x;}
      float value(){return float(next()>>8)*(1.f/16777216.f);} int integer(int bound){const uint32_t b=uint32_t(bound),th=uint32_t(-b)%b;uint32_t x;do{x=next();}while(x<th);return int(x%b);} } r{s.rng};
    Decision d = update(s, in, cat, r);
    int accepted = -1;
    if (d.move >= 0) { accepted = (rnd() % 4) != 0; attack_result(s, in, accepted != 0); if (accepted) punchLeft = 20 + rnd() % 140; }
    if (punchLeft > 0) punchLeft--;
    Command c = locomotion(s, in);
    printf("{\"n\":%d,\"dist\":%.9g,\"ang\":%.9g,\"dt\":%.9g,\"t\":%.9g,\"punching\":%d,\"down\":%d,\"active\":%d,"
           "\"phase\":%d,\"timer\":%.9g,\"move\":%d,\"accepted\":%d,\"clear\":%d,\"f\":%.9g,\"s\":%.9g,\"y\":%.9g,\"rng\":%u}\n",
           n, dist, ang, dt, t, int(in.punching), int(down), int(in.round_active), s.phase, s.timer, d.move, accepted,
           int(d.clear_punching), c.forward, c.strafe, c.yaw, s.rng);
  }
}
