// Fixed-capacity output, live-slot contact bookkeeping. No physics/scoring.
#include <cstdint>
#include <cmath>
#include <cstddef>
#ifdef __CUDACC__
#include <cuda_runtime.h>
#define KERNEL __global__
#define DEVICE __device__
#define HOST_DEVICE __host__ __device__
#define EACH(i, n) for (int64_t i = int64_t(blockIdx.x)*blockDim.x+threadIdx.x; i < (n); i += int64_t(blockDim.x)*gridDim.x)
#else
#define KERNEL static
#define DEVICE
#define HOST_DEVICE
#define EACH(i, n) for (int64_t i = 0; i < (n); ++i)
#endif

struct Measurement {
    int32_t arenas, bodies, geoms, capacity, floor;
    const int32_t *geom, *world, *nacon;
    const float *dist, *pos, *frame, *time, *xpos, *xipos, *com, *cvel;
    const int64_t *geom_body, *body_root, *owner, *zone, *part, *side, *slot;
    uint8_t *previous;
    int64_t *expected;
    int32_t *first, *floor_count, *fall_bad, *hit_bad, *candidate_bad, *world_bad;
    uint8_t *capacity_overflow, *fall_valid, *floor_contact, *base_valid, *scan_valid;
    int64_t *integers;
    float *floats;
    uint8_t *candidate_valid;
    int64_t *keys, *counts;
    const float *velocity, *speed;
};

DEVICE void mark(int32_t* p) {
#ifdef __CUDACC__
    atomicExch(p, 1);
#else
    *p = 1;
#endif
}
DEVICE void minimum(int32_t* p, int32_t v) {
#ifdef __CUDACC__
    atomicMin(p, v);
#else
    if (v < *p) *p = v;
#endif
}
DEVICE void count_one(int64_t* p) {
#ifdef __CUDACC__
    atomicAdd(reinterpret_cast<unsigned long long*>(p), 1ull);
#else
    ++*p;
#endif
}
DEVICE bool finite(float x) { return std::isfinite(x); }
HOST_DEVICE int64_t span(Measurement d) { return int64_t(d.geoms)*d.geoms; }
DEVICE int live_count(Measurement d) {
    int n = *d.nacon;
    return n < 0 ? 0 : n > d.capacity ? d.capacity : n;
}
DEVICE int64_t pair_key(Measurement d, int arena, int a, int b) {
    return int64_t(arena)*span(d) + int64_t(a < b ? a : b)*d.geoms + (a > b ? a : b);
}

KERNEL void clear_facts(Measurement d) {
    EACH(i, int64_t(d.arenas)*span(d)) d.first[i] = d.capacity;
    EACH(i, int64_t(d.arenas)*d.bodies) d.floor_count[i] = 0;
    EACH(i, d.arenas) {
        d.fall_bad[i] = d.hit_bad[i] = d.candidate_bad[i] = 0;
        d.counts[i] = 0;
    }
    EACH(i, int64_t(d.capacity)*2) {
        d.candidate_valid[i] = 0;
        d.keys[i] = INT64_MAX;
    }
    EACH(i, 1) {
        *d.world_bad = 0;
        *d.capacity_overflow = *d.nacon < 0 || *d.nacon > d.capacity;
    }
}

KERNEL void contact_facts(Measurement d) {
    EACH(i, live_count(d)) {
        int arena = d.world[i], a = d.geom[2*i], b = d.geom[2*i+1];
        if (arena < 0 || arena >= d.arenas) { mark(d.world_bad); continue; }
        if (a < 0 || a >= d.geoms || b < 0 || b >= d.geoms) {
            mark(d.fall_bad+arena); mark(d.hit_bad+arena); continue;
        }
        if ((a == d.floor) != (b == d.floor)) {
            int64_t body = d.geom_body[a == d.floor ? b : a];
            if (body > 0 && body < d.bodies) mark(d.floor_count+int64_t(arena)*d.bodies+body);
        }
        bool ok = finite(d.dist[i]);
        for (int k = 0; k < 3; ++k) ok = ok && finite(d.pos[3*i+k]);
        for (int k = 0; k < 9; ++k) ok = ok && finite(d.frame[9*i+k]);
        if (a == b || !ok) { mark(d.hit_bad+arena); continue; }
        minimum(d.first+pair_key(d, arena, a, b), int32_t(i));
    }
}

KERNEL void finish_facts(Measurement d) {
    EACH(i, d.arenas) d.fall_valid[i] = !*d.capacity_overflow && !*d.world_bad && !d.fall_bad[i];
    EACH(i, int64_t(d.arenas)*d.bodies) d.floor_contact[i] = d.floor_count[i] != 0;
}

KERNEL void prepare_hits(Measurement d, int substep) {
    EACH(i, d.arenas) d.base_valid[i] = !*d.capacity_overflow && !*d.world_bad
        && !d.hit_bad[i] && d.expected[i] == substep && finite(d.time[i]);
}

KERNEL void pack_hits(Measurement d, int substep) {
    EACH(i, int64_t(live_count(d))*2) {
        int64_t raw = i/2;
        int arena = d.world[raw], a = d.geom[2*raw], b = d.geom[2*raw+1];
        if (arena < 0 || arena >= d.arenas || !d.base_valid[arena]
                || a < 0 || a >= d.geoms || b < 0 || b >= d.geoms) continue;
        int64_t key = pair_key(d, arena, a, b);
        if (d.first[key] != raw || d.previous[key]) continue;
        int sg = i%2 ? b : a, tg = i%2 ? a : b;
        int sb = int(d.geom_body[sg]), tb = int(d.geom_body[tg]);
        int ss = sb < 0 ? 0 : sb >= d.bodies ? d.bodies-1 : sb;
        int ts = tb < 0 ? 0 : tb >= d.bodies ? d.bodies-1 : tb;
        int64_t so = d.owner[ss], to = d.owner[ts], slot = d.slot[ss];
        if (so < 0 || to < 0 || so == to || slot < 0) continue;
        int64_t part = d.part[ss], side = d.side[ss], zone = d.zone[tg];
        if (slot >= 6 || !(part == 1 || part == 2 || part == 8)
                || !(side == 0 || side == 1) || zone < 0 || zone > 17) {
            mark(d.candidate_bad+arena); continue;
        }
        float* out = d.floats+13*i;
        const float* sp = d.xpos+(int64_t(arena)*d.bodies+ss)*3;
        const float* tp = d.xpos+(int64_t(arena)*d.bodies+ts)*3;
        for (int k = 0; k < 3; ++k) { out[k] = sp[k]; out[3+k] = tp[k]; }
        const float* sv = d.velocity+(int64_t(arena)*d.bodies+ss)*3;
        const float* tv = d.velocity+(int64_t(arena)*d.bodies+ts)*3;
        for (int k = 0; k < 3; ++k) { out[6+k] = sv[k]; out[9+k] = tv[k]; }
        bool ok = true;
        for (int k = 0; k < 12; ++k) ok = ok && finite(out[k]);
        if (!ok) { mark(d.candidate_bad+arena); continue; }
        int64_t* in = d.integers+12*i;
        in[0] = arena; in[1] = substep; in[2] = sg; in[3] = tg;
        in[4] = sb; in[5] = tb; in[6] = so; in[7] = to;
        in[8] = slot; in[9] = part; in[10] = side; in[11] = zone;
        d.candidate_valid[i] = 1;
    }
}

KERNEL void validate_speed(Measurement d) {
    EACH(i, int64_t(live_count(d))*2) {
        if (!d.candidate_valid[i]) continue;
        d.floats[13*i+12] = d.speed[i];
        if (!finite(d.speed[i])) mark(d.candidate_bad+int(d.integers[12*i]));
    }
}

KERNEL void finish_hits(Measurement d) {
    EACH(i, int64_t(live_count(d))*2) {
        if (!d.candidate_valid[i]) continue;
        int arena = int(d.integers[12*i]);
        if (d.candidate_bad[arena]) { d.candidate_valid[i] = 0; continue; }
        d.keys[i] = int64_t(arena)*(2*d.capacity)+i;
        count_one(d.counts+arena);
    }
}

KERNEL void commit_hits(Measurement d, int substep) {
    EACH(i, int64_t(d.arenas)*span(d)) {
        int arena = int(i/span(d));
        if (d.base_valid[arena] && !d.candidate_bad[arena]) d.previous[i] = d.first[i] < d.capacity;
    }
    EACH(i, d.arenas) {
        bool valid = d.base_valid[i] && !d.candidate_bad[i];
        d.scan_valid[i] = valid;
        if (valid) d.expected[i] = (substep+1)%10;
    }
}

#ifdef __CUDACC__
static unsigned grid(int64_t n) { return unsigned(n/128+(n%128 != 0)); }
#define LAUNCH(fn, n, ...) fn<<<grid(n),128,0,static_cast<cudaStream_t>(stream)>>>(*d, ##__VA_ARGS__)
#define STATUS int(cudaGetLastError())
#else
#define LAUNCH(fn, n, ...) fn(*d, ##__VA_ARGS__)
#define STATUS 0
#endif

extern "C" size_t rek_measurement_descriptor_size() { return sizeof(Measurement); }
extern "C" int rek_measurement_facts(const Measurement* d, void* stream) {
    if (!d || d->arenas < 1 || d->capacity < 1 || d->bodies < 1 || d->geoms < 1) return -1;
    LAUNCH(clear_facts, int64_t(d->arenas)*span(*d));
    LAUNCH(contact_facts, d->capacity);
    LAUNCH(finish_facts, int64_t(d->arenas)*d->bodies);
    return STATUS;
}
extern "C" int rek_measurement_hits_prepare(const Measurement* d, int substep, void* stream) {
    if (!d || substep < 0 || substep >= 10) return -1;
    LAUNCH(prepare_hits, d->arenas, substep);
    LAUNCH(pack_hits, int64_t(d->capacity)*2, substep);
    return STATUS;
}
extern "C" int rek_measurement_hits_finish(const Measurement* d, int substep, void* stream) {
    if (!d || substep < 0 || substep >= 10) return -1;
    LAUNCH(validate_speed, int64_t(d->capacity)*2);
    LAUNCH(finish_hits, int64_t(d->capacity)*2);
    LAUNCH(commit_hits, int64_t(d->arenas)*span(*d), substep);
    return STATUS;
}
