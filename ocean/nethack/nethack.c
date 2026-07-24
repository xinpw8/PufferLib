#include <time.h>
#include <unistd.h>
#include <string.h>
#include <termios.h>
#include <sys/select.h>
#include <sys/stat.h>
#include <dirent.h>
#include <limits.h>
#include <signal.h>
#include "nethack.h"
#include "../../src/puffercpu.h"
#include "glyph_map.h"

// single-agent env, reset immediately (training's c_reset is lazy)
static void env_open(Nethack* env) {
    memset(env, 0, sizeof(*env));
    env->num_agents = 1;
    env->observations = (unsigned char*)calloc(NETHACK_OBS_SIZE, 1);
    env->actions      = (float*)calloc(14, sizeof(float));   // {verb, 12 per-verb slots, direction}
    env->action_mask  = (unsigned char*)calloc(NETHACK_NUM_ACTIONS
                        + 12 * NETHACK_INV_SLOTS + NETHACK_NUM_DIRS, 1);
    env->rewards      = (float*)calloc(1, sizeof(float));
    env->terminals    = (float*)calloc(1, sizeof(float));
    init(env);
    nethack_do_reset(env);
}

static void env_close(Nethack* env) {
    c_close(env);
    free(env->observations); free(env->actions); free(env->rewards); free(env->terminals);
    free(env->action_mask);
}

// CPU port of the CUDA encoder (src/nethack.cu) + puffernet MinGRU/decoder;
// weight order matches param registration: encoder, decoder, mingru
#define DEMO_VOCAB   5977
#define DEMO_EMBED   32
#define DEMO_BL_FEAT (25 + 7 + 13 + NETHACK_NUM_ACTIONS + NETHACK_NUM_OCLASSES + 2 + 8 + 2)
#define DEMO_INV_HID 16   // 16-dim slot rep: pool bottleneck + decoder key (unified)
#define DEMO_INV_FLAT (NETHACK_INV_SLOTS * DEMO_INV_HID)
#define DEMO_INV_POOL 128
#define DEMO_SFEAT 24    // buc4 + known+spe + quan + ero2 + flags7 + tk + armcat7
#define DEMO_OD (NETHACK_NUM_ACTIONS + 12 * NETHACK_INV_SLOTS + NETHACK_NUM_DIRS)
#define DEMO_NUM_HEADS 14
#define DEMO_PTR_HEADS 12
#define DEMO_QDIM (DEMO_PTR_HEADS * DEMO_INV_HID)
#define DEMO_DEC_PAD 32
#define DEMO_DEC_LIN (NETHACK_NUM_ACTIONS + NETHACK_NUM_DIRS + 1)
#define DEMO_LOC_IN  (NETHACK_CROP_GRID * DEMO_EMBED)   // 9x9 crop, per-cell embeds
#define DEMO_LOC_HID 256
#define DEMO_PW 5
#define DEMO_PH 5
#define DEMO_PX 16
#define DEMO_PY 5
#define DEMO_TOK     (DEMO_PX * DEMO_PY)                // 5x5 patches over 79x21
#define DEMO_PCELLS  (DEMO_PW * DEMO_PH)                // off-map cells read the pad glyph
#define DEMO_P1      16
#define DEMO_GLB_IN  (DEMO_PCELLS * DEMO_EMBED)         // per-patch flatten (glyph slice)
#define DEMO_GLB_HID 128
// trigram message branch, mirroring NH_MSG_* in src/nethack.cu
#define DEMO_MSG_LEN   NETHACK_MSG_LEN
#define DEMO_MSG_VOCAB 4096
#define DEMO_MSG_LOG2V 12
#define DEMO_MSG_HID   32
#define DEMO_MSG_CONCAT_OFF (DEMO_LOC_HID + DEMO_GLB_HID + DEMO_INV_POOL + 64 + DEMO_BL_FEAT)
#define DEMO_CONCAT  (DEMO_MSG_CONCAT_OFF + DEMO_MSG_HID)

// per-blstat normalization, mirroring NH_BL_SCALE / NH_BL_ISLOG in src/nethack.cu
static const float DEMO_BL_SCALE[27] = {
    1.f/79, 1.f/21,
    1.f/25, 1.f/125, 1.f/25, 1.f/25, 1.f/25, 1.f/25, 1.f/25,
    0.1f, 1.f/200, 1.f/200, 1.f/50, 0.1f,
    1.f/100, 1.f/100, 1.f/10, 1.f/10, 1.f/30,
    0.1f, 0.1f, 0.f, 1.f/4, 0.f, 1.f/50, 0.f, 1.f,   // dnum one-hot (scale dead)
};
static const int DEMO_BL_ISLOG[27] =
    {0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,1,1,0,0,0,0,0,0};

typedef struct {
    float *embed;               // (5977, 32) E_res
    float *ekind_w, *esub_w;    // (14, 32), (944, 32) factor tables
    float *e_eff;               // materialized E_res + E_kind + E_sub
    float *loc_w, *loc_b;       // (256, 2592), (256)
    float *g1_w, *g1_xy, *g1_b; // (16, 800), (16, 2), (16): per-patch embed+flatten + hero dx,dy -> 16
    float *g2_w, *g2_b;         // (128, 16), (128): 16 -> 128, maxed over tokens
    float *inv1_w, *inv1_b;     // (16, 32), (16): per-slot features (pointer keys)
    float *inv1s_w;             // (16, 24): gated item-state path into the slot MLP
    float *inv2_w, *inv2_b;     // (128, 16), (128): pooled trunk summary (max over slots)
    float *bl_w, *bl_b;         // (64, DEMO_BL_FEAT), (64)
    float *proj_w, *proj_b;     // (H, DEMO_CONCAT), (H)
    float *msg_w;               // (4096, 32) trigram embedding table
    float *dec_lin;             // (32, H) bias-free; rows [22 verb | 8 dir | value], 31 used
    float *dec_q;               // (192, H): twelve stacked 16-dim query projections
    float *dec_k;               // (16, 16): key projection over slot features
    float *dec_tau;             // (12,): per-head log cosine temperature
    MinGRU* mingru;
    Multidiscrete* md;
    int hidden_size, num_layers, num_actions;
    float x[DEMO_LOC_IN];       // crop cell embeds, flattened
    float px[DEMO_GLB_IN];      // one patch's cell embeds, flattened
    float t16[DEMO_P1];
    float t128[DEMO_GLB_HID];
    float slots[DEMO_INV_FLAT]; // per-slot post-relu features (decoder keys)
    float concat[DEMO_CONCAT];  // [local hid | global hid | inv pool | bl hidden | bl feats | msg]
    float logits[DEMO_OD + 1];  // assembled decoder output; last entry is value
    float* hidden;              // (hidden_size)
} NethackNet;

// (hidden, layers) from the checkpoint float count:
//   total = ENC_FIXED + H*(DEMO_CONCAT + 1) + H*(32 + 192) + DEC_FIXED + L * 3*H*H
// All tensors land on 8-float boundaries; only tau (12) needs padding (+4).
#define DEMO_ENC_FIXED (DEMO_VOCAB*DEMO_EMBED \
                        + NH_GM_NKIND*DEMO_EMBED + NH_GM_NSUB*DEMO_EMBED \
                        + DEMO_LOC_HID*DEMO_LOC_IN + DEMO_LOC_HID \
                        + DEMO_P1*DEMO_GLB_IN + DEMO_P1*2 + DEMO_P1 \
                        + DEMO_GLB_HID*DEMO_P1 + DEMO_GLB_HID \
                        + DEMO_INV_HID*DEMO_EMBED + DEMO_INV_HID \
                        + DEMO_INV_HID*DEMO_SFEAT \
                        + DEMO_INV_POOL*DEMO_INV_HID + DEMO_INV_POOL \
                        + 64*DEMO_BL_FEAT + 64 \
                        + DEMO_MSG_VOCAB*DEMO_MSG_HID)
#define DEMO_DEC_FIXED (DEMO_INV_HID*DEMO_INV_HID + 16)   // k_w + tau padded 12->16
// ambiguities are possible; prefer the fewest layers (real configs have <= 8)
static int demo_infer_arch(int total, int* hidden, int* layers, int* actions) {
    int best_l = 1 << 30;
    for (int H = 8; H <= 4096; H += 8) {
        long rem = (long)total - DEMO_ENC_FIXED - DEMO_DEC_FIXED
                 - (long)H * (DEMO_CONCAT + 1 + DEMO_DEC_PAD + DEMO_QDIM);
        long per_layer = 3L * H * H;
        if (rem <= 0) break;
        if (rem % per_layer) continue;
        long L = rem / per_layer;
        if (L >= 1 && L < best_l) { best_l = (int)L; *hidden = H; *layers = (int)L; *actions = NETHACK_NUM_ACTIONS; }
    }
    return best_l == 1 << 30 ? -1 : 0;
}

static NethackNet* make_nethack_net(Weights* w) {
    NethackNet* net = (NethackNet*)calloc(1, sizeof(NethackNet));
    if (demo_infer_arch(w->size - 7, &net->hidden_size, &net->num_layers, &net->num_actions) != 0) {
        fprintf(stderr, "nethack demo: cannot infer arch from %d floats — "
                "checkpoint is not a nethack policy with %d actions?\n",
                w->size - 7, NETHACK_NUM_ACTIONS);
        exit(1);
    }
    fprintf(stderr, "nethack demo: hidden=%d layers=%d actions=%d (%d floats)\n",
            net->hidden_size, net->num_layers, net->num_actions, w->size - 7);
    net->hidden = (float*)calloc(net->hidden_size, sizeof(float));
    net->embed   = get_weights_aligned(w, DEMO_VOCAB * DEMO_EMBED);
    net->ekind_w = get_weights_aligned(w, NH_GM_NKIND * DEMO_EMBED);
    net->esub_w  = get_weights_aligned(w, NH_GM_NSUB * DEMO_EMBED);
    net->loc_w   = get_weights_aligned(w, DEMO_LOC_HID * DEMO_LOC_IN);
    net->loc_b   = get_weights_aligned(w, DEMO_LOC_HID);
    net->g1_w    = get_weights_aligned(w, DEMO_P1 * DEMO_GLB_IN);
    net->g1_xy   = get_weights_aligned(w, DEMO_P1 * 2);
    net->g1_b    = get_weights_aligned(w, DEMO_P1);
    net->g2_w    = get_weights_aligned(w, DEMO_GLB_HID * DEMO_P1);
    net->g2_b    = get_weights_aligned(w, DEMO_GLB_HID);
    net->inv1_w  = get_weights_aligned(w, DEMO_INV_HID * DEMO_EMBED);
    net->inv1_b  = get_weights_aligned(w, DEMO_INV_HID);
    net->inv1s_w = get_weights_aligned(w, DEMO_INV_HID * DEMO_SFEAT);
    net->inv2_w  = get_weights_aligned(w, DEMO_INV_POOL * DEMO_INV_HID);
    net->inv2_b  = get_weights_aligned(w, DEMO_INV_POOL);
    net->bl_w    = get_weights_aligned(w, 64 * DEMO_BL_FEAT);
    net->bl_b    = get_weights_aligned(w, 64);
    net->proj_w  = get_weights_aligned(w, net->hidden_size * DEMO_CONCAT);
    net->proj_b  = get_weights_aligned(w, net->hidden_size);
    net->msg_w   = get_weights_aligned(w, DEMO_MSG_VOCAB * DEMO_MSG_HID);
    net->dec_lin = get_weights_aligned(w, DEMO_DEC_PAD * net->hidden_size);
    net->dec_q   = get_weights_aligned(w, DEMO_QDIM * net->hidden_size);
    net->dec_k   = get_weights_aligned(w, DEMO_INV_HID * DEMO_INV_HID);
    net->dec_tau = get_weights_aligned(w, DEMO_PTR_HEADS);
    net->mingru  = make_mingru(w, 1, net->hidden_size, net->num_layers);
    static int logit_sizes[DEMO_NUM_HEADS] = {
        NETHACK_NUM_ACTIONS, NETHACK_INV_SLOTS, NETHACK_INV_SLOTS, NETHACK_INV_SLOTS,
        NETHACK_INV_SLOTS, NETHACK_INV_SLOTS, NETHACK_INV_SLOTS, NETHACK_INV_SLOTS,
        NETHACK_INV_SLOTS, NETHACK_INV_SLOTS, NETHACK_INV_SLOTS, NETHACK_INV_SLOTS,
        NETHACK_INV_SLOTS, NETHACK_NUM_DIRS};
    net->md = make_multidiscrete(1, logit_sizes, DEMO_NUM_HEADS);
    assert(w->idx == w->size - 7);
    // materialize the residual-factorized embedding once (host, load time)
    net->e_eff = (float*)malloc((size_t)DEMO_VOCAB * DEMO_EMBED * sizeof(float));
    for (int g = 0; g < DEMO_VOCAB; g++)
        for (int d = 0; d < DEMO_EMBED; d++)
            net->e_eff[g * DEMO_EMBED + d] = net->embed[g * DEMO_EMBED + d]
                + net->ekind_w[nh_glyph_kind[g] * DEMO_EMBED + d]
                + net->esub_w[nh_glyph_sub[g] * DEMO_EMBED + d];
    return net;
}

static inline int demo_msg_lc(int c) {
    return (c >= 'A' && c <= 'Z') ? c + 32 : c;   // lowercase; keep spaces/punct
}
static inline int demo_msg_hash(int c0, int c1, int c2) {
    unsigned key = ((unsigned)c0 << 16) | ((unsigned)c1 << 8) | (unsigned)c2;
    return (int)((key * 2654435761u) >> (32 - DEMO_MSG_LOG2V));
}
// normalized-sum trigram bag over the null-terminated topline; scaled by
// 1/sqrt(count+1), no relu (raw signed summary)
static void demo_msg_pool(NethackNet* net, const unsigned char* obs, float* out) {
    const unsigned char* m = obs + NETHACK_OFF_MSG;
    for (int d = 0; d < DEMO_MSG_HID; d++) out[d] = 0.0f;
    int count = 0;
    for (int t = 0; t <= DEMO_MSG_LEN - 3; t++) {
        int c0 = m[t], c1 = m[t + 1], c2 = m[t + 2];
        if (c0 == 0 || c1 == 0 || c2 == 0) break;
        int id = demo_msg_hash(demo_msg_lc(c0), demo_msg_lc(c1), demo_msg_lc(c2));
        count++;
        for (int d = 0; d < DEMO_MSG_HID; d++)
            out[d] += net->msg_w[(size_t)id * DEMO_MSG_HID + d];
    }
    float scale = 1.0f / sqrtf((float)count + 1.0f);
    for (int d = 0; d < DEMO_MSG_HID; d++) out[d] *= scale;
}

// blstats/extra live at unaligned byte offsets: assemble, don't cast
static int32_t demo_i32(const unsigned char* p) {
    int32_t v;
    memcpy(&v, p, 4);
    return v;
}

static int demo_glyph_at(const int16_t* glyphs, int r, int c) {
    if (r < 0 || r >= NH_ROWS || c < 0 || c >= NH_COLS) return NETHACK_PAD_GLYPH;
    int g = glyphs[r * NH_COLS + c];
    if (g < 0) g = 0;
    if (g >= DEMO_VOCAB) g = DEMO_VOCAB - 1;
    return g;
}

static int nethack_net_forward(NethackNet* net, const unsigned char* obs) {   // fills decoder->output
    const int16_t* glyphs = (const int16_t*)(obs + NETHACK_OFF_GLYPHS);
    const unsigned char* bl = obs + NETHACK_OFF_BLSTATS;

    // local view: per-cell embeds of the egocentric crop, flattened
    int hx = demo_i32(bl), hy = demo_i32(bl + 4);
    int half = NETHACK_CROP / 2;
    for (int p = 0; p < NETHACK_CROP_GRID; p++) {
        int g = demo_glyph_at(glyphs, hy - half + p / NETHACK_CROP,
                              hx - half + p % NETHACK_CROP);
        memcpy(net->x + p * DEMO_EMBED, net->e_eff + g * DEMO_EMBED,
               DEMO_EMBED * sizeof(float));
    }
    _linear(net->x, net->loc_w, net->loc_b, net->concat, 1, DEMO_LOC_IN, DEMO_LOC_HID);
    _relu(net->concat, net->concat, DEMO_LOC_HID);

    // global view: per patch embed+flatten + normalized hero (dx,dy) -> 16 ->
    // 128, elementwise max over the 80 tokens (off-map cells of ragged edge
    // patches read the pad glyph)
    float* glb = net->concat + DEMO_LOC_HID;
    for (int o = 0; o < DEMO_GLB_HID; o++) glb[o] = -1e30f;
    for (int tk = 0; tk < DEMO_TOK; tk++) {
        int r0 = (tk / DEMO_PX) * DEMO_PH, c0 = (tk % DEMO_PX) * DEMO_PW;
        for (int pos = 0; pos < DEMO_PCELLS; pos++) {
            int g = demo_glyph_at(glyphs, r0 + pos / DEMO_PW, c0 + pos % DEMO_PW);
            memcpy(net->px + pos * DEMO_EMBED, net->e_eff + g * DEMO_EMBED,
                   DEMO_EMBED * sizeof(float));
        }
        float dx = (c0 + 0.5f * (DEMO_PW - 1) - hx) / (float)NH_COLS;
        float dy = (r0 + 0.5f * (DEMO_PH - 1) - hy) / (float)NH_ROWS;
        _linear(net->px, net->g1_w, net->g1_b, net->t16, 1, DEMO_GLB_IN, DEMO_P1);
        for (int k = 0; k < DEMO_P1; k++) {
            net->t16[k] += dx * net->g1_xy[k * 2] + dy * net->g1_xy[k * 2 + 1];
            if (net->t16[k] < 0.f) net->t16[k] = 0.f;
        }
        _linear(net->t16, net->g2_w, net->g2_b, net->t128, 1, DEMO_P1, DEMO_GLB_HID);
        for (int o = 0; o < DEMO_GLB_HID; o++)
            if (net->t128[o] > glb[o]) glb[o] = net->t128[o];
    }
    _relu(glb, glb, DEMO_GLB_HID);

    // inventory entities: per-slot embed -> shared 32->32 linear+relu (kept
    // as the pointer decoder's keys), then 32 -> 128 with max over slots for
    // the trunk (matches the CUDA fused pool)
    const int16_t* inv = (const int16_t*)(obs + NETHACK_OFF_INV);
    const signed char* invst = (const signed char*)(obs + NETHACK_OFF_INVST);
    for (int slot = 0; slot < NETHACK_INV_SLOTS; slot++) {
        int g = inv[slot];
        if (g < 0) g = 0;
        if (g >= DEMO_VOCAB) g = DEMO_VOCAB - 1;
        const signed char* st = invst + slot * NLE_INV_STATE_FIELDS;
        float sf[DEMO_SFEAT];
        for (int c = 0; c < 4; c++) sf[c] = st[0] == c ? 1.0f : 0.0f;
        int spe_known = st[1] != -128;
        sf[4] = (float)spe_known;
        sf[5] = spe_known ? (float)st[1] * 0.1f : 0.0f;
        sf[6] = log1pf(fmaxf((float)st[2], 0.0f)) * 0.5f;
        sf[7] = (float)st[3] * (1.0f / 3.0f);
        sf[8] = (float)st[4] * (1.0f / 3.0f);
        for (int c = 0; c < 7; c++) sf[9 + c] = (float)((st[5] >> c) & 1);
        sf[16] = (float)st[6];
        int ot = inv[slot] - NH_GLYPH_OBJ_OFF;   // armor slot category one-hot
        int cat = (ot >= 0 && ot < NH_NUM_OBJECTS) ? nh_obj_armcat[ot] : -1;
        for (int c = 0; c < 7; c++) sf[17 + c] = cat == c ? 1.0f : 0.0f;
        float* h32 = net->slots + slot * DEMO_INV_HID;
        _linear(net->e_eff + g * DEMO_EMBED, net->inv1_w, net->inv1_b,
                h32, 1, DEMO_EMBED, DEMO_INV_HID);
        for (int k = 0; k < DEMO_INV_HID; k++)
            for (int j = 0; j < DEMO_SFEAT; j++)
                h32[k] += net->inv1s_w[k * DEMO_SFEAT + j] * sf[j];
        _relu(h32, h32, DEMO_INV_HID);
    }
    float* invp = net->concat + DEMO_LOC_HID + DEMO_GLB_HID;
    for (int o = 0; o < DEMO_INV_POOL; o++) {
        float best = -1e30f;
        for (int slot = 0; slot < NETHACK_INV_SLOTS; slot++) {
            float v = 0.0f;
            for (int k = 0; k < DEMO_INV_HID; k++)
                v += net->inv2_w[o * DEMO_INV_HID + k] * net->slots[slot * DEMO_INV_HID + k];
            if (v > best) best = v;
        }
        invp[o] = fmaxf(best + net->inv2_b[o], 0.0f);
    }

    // blstats+extra features (25 scalars, hunger 7, cond bits 13, prev verb
    // one-hot, inv class counts, hp/ene frac, dnum one-hot, engraving bits)
    float* f = net->concat + DEMO_LOC_HID + DEMO_GLB_HID + DEMO_INV_POOL + 64;
    int j = 0;
    for (int i = 0; i < 27; i++) {
        if (i == 21 || i == 25) continue;   // hunger, condition: expanded below
        float v = (float)demo_i32(bl + 4*i);
        f[j++] = DEMO_BL_ISLOG[i] ? log1pf(fmaxf(v, 0.f)) * DEMO_BL_SCALE[i]
                                  : v * DEMO_BL_SCALE[i];
    }
    int h21 = demo_i32(bl + 4*21);
    int hunger = h21 < 0 ? 0 : (h21 > 6 ? 6 : h21);
    for (int h = 0; h < 7; h++) f[j++] = (h == hunger) ? 1.f : 0.f;
    for (int k = 0; k < 13; k++) f[j++] = (float)(((uint32_t)demo_i32(bl + 4*25) >> k) & 1u);
    const unsigned char* ex = obs + NETHACK_OFF_EXTRA;
    for (int h = 0; h < NETHACK_NUM_ACTIONS; h++) f[j++] = (h == demo_i32(ex + 4)) ? 1.f : 0.f;
    for (int k = 0; k < NETHACK_NUM_OCLASSES; k++) f[j++] = (float)demo_i32(ex + 4*(2 + k)) * 0.125f;
    for (int p = 0; p < 2; p++) {   // hp_frac, ene_frac
        int cur = demo_i32(bl + 4*(p ? 14 : 10)), mx = demo_i32(bl + 4*(p ? 15 : 11));
        f[j++] = fminf(fmaxf((float)cur / (float)(mx > 1 ? mx : 1), 0.f), 1.f);
    }
    int d23 = demo_i32(bl + 4*23);
    int dnum = d23 < 0 ? 0 : (d23 > 7 ? 7 : d23);
    for (int d = 0; d < 8; d++) f[j++] = (d == dnum) ? 1.f : 0.f;
    int engr = demo_i32(ex);
    f[j++] = engr >= 1 ? 1.f : 0.f;   // any engraving underfoot
    f[j++] = engr >= 2 ? 1.f : 0.f;   // active Elbereth
    for (int k = 0; k < DEMO_BL_FEAT; k++) f[k] = fminf(fmaxf(f[k], -1.f), 1.f);

    float* blout = net->concat + DEMO_LOC_HID + DEMO_GLB_HID + DEMO_INV_POOL;
    _linear(f, net->bl_w, net->bl_b, blout, 1, DEMO_BL_FEAT, 64);
    _relu(blout, blout, 64);

    demo_msg_pool(net, obs, net->concat + DEMO_MSG_CONCAT_OFF);

    _linear(net->concat, net->proj_w, net->proj_b, net->hidden, 1, DEMO_CONCAT, net->hidden_size);
    _relu(net->hidden, net->hidden, net->hidden_size);

    mingru(net->mingru, net->hidden);

    // pointer decoder: [22 verb | 12x55 slots | 8 dir | value]. verb/dir/value
    // from one bias-free linear; slot logit i = tau_h * cos(q_h, k_i) with
    // keys k_i projected from the per-slot features above.
    float* hs = net->mingru->output;
    int H = net->hidden_size;
    float tmp[DEMO_DEC_LIN];
    for (int r = 0; r < DEMO_DEC_LIN; r++) {
        float acc = 0.0f;
        for (int k = 0; k < H; k++) acc += net->dec_lin[r * H + k] * hs[k];
        tmp[r] = acc;
    }
    float q[DEMO_QDIM];
    for (int r = 0; r < DEMO_QDIM; r++) {
        float acc = 0.0f;
        for (int k = 0; k < H; k++) acc += net->dec_q[r * H + k] * hs[k];
        q[r] = acc;
    }
    float kmat[DEMO_INV_FLAT], kn[NETHACK_INV_SLOTS];
    for (int i = 0; i < NETHACK_INV_SLOTS; i++) {
        float nk = 0.0f;
        for (int r = 0; r < DEMO_INV_HID; r++) {
            float acc = 0.0f;
            for (int k = 0; k < DEMO_INV_HID; k++)
                acc += net->dec_k[r * DEMO_INV_HID + k] * net->slots[i * DEMO_INV_HID + k];
            kmat[i * DEMO_INV_HID + r] = acc;
            nk += acc * acc;
        }
        kn[i] = sqrtf(nk) + 1e-6f;
    }
    for (int a = 0; a < NETHACK_NUM_ACTIONS; a++) net->logits[a] = tmp[a];
    for (int h = 0; h < DEMO_PTR_HEADS; h++) {
        const float* qh = q + h * DEMO_INV_HID;
        float nq = 0.0f;
        for (int k = 0; k < DEMO_INV_HID; k++) nq += qh[k] * qh[k];
        nq = sqrtf(nq) + 1e-6f;
        for (int i = 0; i < NETHACK_INV_SLOTS; i++) {
            float dot = 0.0f;
            for (int k = 0; k < DEMO_INV_HID; k++)
                dot += qh[k] * kmat[i * DEMO_INV_HID + k];
            net->logits[NETHACK_NUM_ACTIONS + h * NETHACK_INV_SLOTS + i] =
                expf(net->dec_tau[h]) * dot / (nq * kn[i]);
        }
    }
    for (int d = 0; d <= NETHACK_NUM_DIRS; d++)   // 8 dirs + value
        net->logits[NETHACK_NUM_ACTIONS + DEMO_PTR_HEADS * NETHACK_INV_SLOTS + d] =
            tmp[NETHACK_NUM_ACTIONS + d];
    return 0;
}

// ---- interactive TTY demo -------------------------------------------------
// Space: one step on press; hold advances at 5 Hz. Shift+Space (or hold S): 20 Hz.
// Terminals that support xterm modifyOtherKeys report Shift+Space as a CSI
// sequence; 'S' is the fallback for everything else. q / Esc quits.

static struct termios g_term_orig;
static int g_term_raw = 0;

static void demo_restore_term(void) {
    if (!g_term_raw) return;
    printf("\x1b[>4;0m");           // disable modifyOtherKeys
    printf("\x1b[?25h");            // show cursor
    tcsetattr(STDIN_FILENO, TCSAFLUSH, &g_term_orig);
    g_term_raw = 0;
    fflush(stdout);
}

static void demo_on_signal(int sig) {
    (void)sig;
    demo_restore_term();
    _exit(128 + sig);
}

static void demo_raw_term(void) {
    if (!isatty(STDIN_FILENO)) return;
    if (tcgetattr(STDIN_FILENO, &g_term_orig) != 0) return;
    atexit(demo_restore_term);
    signal(SIGINT, demo_on_signal);
    signal(SIGTERM, demo_on_signal);
    struct termios t = g_term_orig;
    t.c_lflag &= (tcflag_t)~(ICANON | ECHO);
    t.c_cc[VMIN] = 0;
    t.c_cc[VTIME] = 0;
    tcsetattr(STDIN_FILENO, TCSAFLUSH, &t);
    // modifyOtherKeys level 2: Shift+Space -> CSI 27;2;32~
    printf("\x1b[>4;2m\x1b[?25l");
    fflush(stdout);
    g_term_raw = 1;
}

static double demo_now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

// Weight resolution (eval the policy you just trained, not a stale fallback):
//   1. NH_WEIGHTS if set
//   2. Most recently written run under checkpoints/nethack/*/, highest-step
//      .bin in that run (matches the intent of puffer eval --load-model-path=latest
//      without being poisoned by an older multi-B-step run for a different role)
//   3. resources/nethack/nethack_weights.bin (checked-in demo fallback only)
// Global highest-step is wrong after a role switch (e.g. monk 2B vs valk 200M):
// the bigger step number is a different character. Resources-first is worse:
// a committed demo bin silently masks every local training run.
static const char* demo_find_weights(void) {
    const char* envw = getenv("NH_WEIGHTS");
    if (envw && envw[0]) return envw;

    static const char* resources = "resources/nethack/nethack_weights.bin";
    static char best_path[PATH_MAX];
    best_path[0] = '\0';

    // Pass 1: find the run directory with the newest .bin mtime.
    char best_run[PATH_MAX];
    best_run[0] = '\0';
    time_t best_run_mt = 0;
    DIR* root = opendir("checkpoints/nethack");
    if (root) {
        struct dirent* run;
        while ((run = readdir(root)) != NULL) {
            if (run->d_name[0] == '.') continue;
            char rundir[PATH_MAX];
            int n = snprintf(rundir, sizeof(rundir), "checkpoints/nethack/%s", run->d_name);
            if (n < 0 || n >= (int)sizeof(rundir)) continue;
            DIR* rd = opendir(rundir);
            if (!rd) continue;
            time_t run_mt = 0;
            struct dirent* f;
            while ((f = readdir(rd)) != NULL) {
                size_t len = strlen(f->d_name);
                if (len < 5 || strcmp(f->d_name + len - 4, ".bin") != 0) continue;
                char path[PATH_MAX];
                n = snprintf(path, sizeof(path), "%s/%s", rundir, f->d_name);
                if (n < 0 || n >= (int)sizeof(path)) continue;
                struct stat st;
                if (stat(path, &st) != 0 || !S_ISREG(st.st_mode)) continue;
                if (st.st_mtime >= run_mt) run_mt = st.st_mtime;
            }
            closedir(rd);
            if (run_mt > best_run_mt) {
                best_run_mt = run_mt;
                snprintf(best_run, sizeof(best_run), "%s", rundir);
            }
        }
        closedir(root);
    }

    // Pass 2: highest zero-padded step within that run.
    if (best_run[0]) {
        long long best_step = -1;
        time_t best_mt = 0;
        DIR* rd = opendir(best_run);
        if (rd) {
            struct dirent* f;
            while ((f = readdir(rd)) != NULL) {
                size_t len = strlen(f->d_name);
                if (len < 5 || strcmp(f->d_name + len - 4, ".bin") != 0) continue;
                char path[PATH_MAX];
                int n = snprintf(path, sizeof(path), "%s/%s", best_run, f->d_name);
                if (n < 0 || n >= (int)sizeof(path)) continue;
                struct stat st;
                if (stat(path, &st) != 0 || !S_ISREG(st.st_mode)) continue;
                long long step = 0;
                int digits = 0;
                for (const char* p = f->d_name; *p >= '0' && *p <= '9'; p++) {
                    step = step * 10 + (*p - '0');
                    digits++;
                }
                if (digits == 0) step = (long long)st.st_mtime;
                if (step > best_step || (step == best_step && st.st_mtime >= best_mt)) {
                    best_step = step;
                    best_mt = st.st_mtime;
                    snprintf(best_path, sizeof(best_path), "%s", path);
                }
            }
            closedir(rd);
        }
    }
    if (best_path[0]) return best_path;
    return resources;
}

// Drain stdin. Returns a bitset: bit0=space, bit1=shift+space/S, bit2=quit.
// Hold is detected via OS auto-repeat (and CSI for Shift+Space).
#define DEMO_IN_SPACE  1
#define DEMO_IN_FAST   2
#define DEMO_IN_QUIT   4

// xterm/kitty encode modifiers as 1 + bitmask (Shift=1, Alt=2, Ctrl=4, ...)
static int demo_mod_shift(int mod_param) {
    int bits = mod_param > 0 ? mod_param - 1 : 0;
    return (bits & 1) != 0;
}

static int demo_poll_input(void) {
    int flags = 0;
    for (;;) {
        fd_set rfds;
        FD_ZERO(&rfds);
        FD_SET(STDIN_FILENO, &rfds);
        struct timeval tv = {0, 0};
        if (select(STDIN_FILENO + 1, &rfds, NULL, NULL, &tv) <= 0) break;
        unsigned char buf[64];
        ssize_t n = read(STDIN_FILENO, buf, sizeof(buf));
        if (n <= 0) break;
        for (ssize_t i = 0; i < n; i++) {
            unsigned char c = buf[i];
            if (c == 'q' || c == 'Q') {
                flags |= DEMO_IN_QUIT;
            } else if (c == 0x1b) {
                // Esc alone (no following bytes in this read) => quit. If CSI,
                // parse modifyOtherKeys / kitty sequences for Space.
                if (i + 1 >= n || buf[i + 1] != '[') { flags |= DEMO_IN_QUIT; continue; }
                i++; // at '['
                int params[8], np = 0, val = 0, in_num = 0;
                memset(params, 0, sizeof(params));
                for (i++; i < n; i++) {
                    unsigned char d = buf[i];
                    if (d >= '0' && d <= '9') { val = val * 10 + (d - '0'); in_num = 1; }
                    else if (d == ';') {
                        if (np < 8) params[np++] = in_num ? val : 0;
                        val = 0; in_num = 0;
                    } else if (d >= 0x40 && d <= 0x7e) {
                        if (in_num && np < 8) params[np++] = val;
                        // xterm modifyOtherKeys: CSI 27 ; mod ; keycode ~
                        if (d == '~' && np >= 3 && params[0] == 27 && params[2] == 32)
                            flags |= demo_mod_shift(params[1]) ? DEMO_IN_FAST : DEMO_IN_SPACE;
                        // kitty CSI u: CSI 32 ; mod u
                        if (d == 'u' && np >= 1 && params[0] == 32) {
                            int mod = np >= 2 ? params[1] : 1;
                            flags |= demo_mod_shift(mod) ? DEMO_IN_FAST : DEMO_IN_SPACE;
                        }
                        break;
                    } else break;
                }
            } else if (c == ' ') {
                flags |= DEMO_IN_SPACE;
            } else if (c == 'S' || c == 's') {
                // fallback fast key when the terminal does not report Shift+Space
                flags |= DEMO_IN_FAST;
            }
        }
    }
    return flags;
}

static void demo_step_once(NethackNet* net, Nethack* env, float* acts_f,
                           float* ep_score, float* ep_len, float* ep_depth,
                           float* ep_xp, float* ep_gt) {
    nethack_net_forward(net, env->observations);
    for (int i = 0; i < DEMO_OD; i++)
        if (!env->action_mask[i]) net->logits[i] = -1e9f;
    softmax_multidiscrete(net->md, net->logits, acts_f);
    for (int h = 0; h < DEMO_NUM_HEADS; h++) env->actions[h] = acts_f[h];
    c_step(env);
    if (env->terminals[0] > 0.5f) {
        float d = env->log.max_depth - *ep_depth;
        float x = env->log.max_xp_level - *ep_xp;
        float g = env->log.game_time - *ep_gt;
        fprintf(stderr, "episode end: score=%.0f len=%.0f max_depth=%.0f xp=%.0f game_t=%.0f\n",
                env->log.score - *ep_score, env->log.episode_length - *ep_len, d, x, g);
        *ep_score = env->log.score;
        *ep_len = env->log.episode_length;
        *ep_depth = env->log.max_depth;
        *ep_xp = env->log.max_xp_level;
        *ep_gt = env->log.game_time;
        memset(net->mingru->state, 0,
               (size_t)net->num_layers * net->hidden_size * sizeof(float));
    }
}

static void demo_render(Nethack* env, int rate_hz, long steps) {
    c_render(env);
    printf("steps %ld  |  SPACE step/hold 5Hz  |  Shift+SPACE (or S) 20Hz  |  q quit",
           steps);
    if (rate_hz > 0) printf("  |  running %d Hz", rate_hz);
    printf("\n");
    fflush(stdout);
}

// Interactive: press space = 1 action; hold = 5/s; shift+space (or S) = 20/s.
static void run_demo_interactive(long max_steps) {
    const char* wpath = demo_find_weights();
    fprintf(stderr, "nethack demo: weights=%s\n", wpath);
    Weights* w = load_weights((char*)wpath);
    if (!w) {
        fprintf(stderr, "nethack demo: %s missing (set NH_WEIGHTS=path/to.bin)\n", wpath);
        exit(1);
    }
    NethackNet* net = make_nethack_net(w);

    Nethack env;
    env_open(&env);
    const char* seed_env = getenv("NH_SEED");
    srand(seed_env ? (unsigned)strtoul(seed_env, NULL, 10) : (unsigned)time(NULL));

    demo_raw_term();

    float ep_score = 0, ep_len = 0, ep_depth = 0, ep_xp = 0, ep_gt = 0;
    float acts_f[DEMO_NUM_HEADS];
    long steps = 0;

    // Hold model (TTY has no key-up; OS auto-repeat confirms a hold):
    //   first SPACE/S  -> exactly one step
    //   further events -> continuous 5 Hz (space) or 20 Hz (shift+space / S)
    // Grace after first press covers the typical OS key-repeat delay (~0.5 s).
    int mode = 0;               // 0=idle, 1=slow, 2=fast
    int confirmed_hold = 0;     // saw a second key event (auto-repeat)
    int edge_pending = 0;
    double held_until = 0;
    double next_step_at = 0;

    demo_render(&env, 0, steps);

    while (steps < max_steps) {
        int in = demo_poll_input();
        if (in & DEMO_IN_QUIT) break;

        double now = demo_now();
        int want = 0;
        if (in & DEMO_IN_FAST) want = 2;
        else if (in & DEMO_IN_SPACE) want = 1;

        if (want) {
            if (mode != want) {
                // new press (or speed change): one immediate step, wait for
                // auto-repeat before continuous advance
                mode = want;
                confirmed_hold = 0;
                edge_pending = 1;
                next_step_at = now;
                held_until = now + 0.55;
            } else {
                // same mode again => key is being held
                confirmed_hold = 1;
                held_until = now + 0.12;
            }
        } else if (mode && now > held_until) {
            mode = 0;
            confirmed_hold = 0;
            edge_pending = 0;
        }

        int do_step = 0;
        int rate = mode == 2 ? 20 : (mode == 1 ? 5 : 0);
        if (mode && edge_pending) {
            do_step = 1;
            edge_pending = 0;
            next_step_at = now + 1.0 / (double)rate;
        } else if (mode && confirmed_hold && now >= next_step_at) {
            do_step = 1;
            next_step_at = now + 1.0 / (double)rate;
        }

        if (do_step) {
            demo_step_once(net, &env, acts_f, &ep_score, &ep_len,
                           &ep_depth, &ep_xp, &ep_gt);
            steps++;
            demo_render(&env, confirmed_hold ? rate : 0, steps);
        } else {
            usleep(5000);
        }
    }

    demo_restore_term();
    if (env.log.n > 0)
        printf("episodes=%.0f  avg_score=%.1f  avg_max_depth=%.2f  avg_xp=%.2f  steps=%ld\n",
               env.log.n, env.log.score / env.log.n,
               env.log.max_depth / env.log.n, env.log.max_xp_level / env.log.n, steps);
    else
        printf("steps=%ld\n", steps);
    env_close(&env);
    free_mingru(net->mingru);
    free(net->md); free(net->hidden); free(net->e_eff); free(net);
    free(w);
}

// Auto-run (headless or fixed frame delay) — used by scripts / profiling.
static void run_demo_auto(long max_steps, int frame_ms) {
    const char* wpath = demo_find_weights();
    fprintf(stderr, "nethack demo: weights=%s\n", wpath);
    Weights* w = load_weights((char*)wpath);
    if (!w) {
        fprintf(stderr, "nethack demo: %s missing (set NH_WEIGHTS=path/to.bin)\n", wpath);
        exit(1);
    }
    NethackNet* net = make_nethack_net(w);

    Nethack env;
    env_open(&env);
    const char* seed_env = getenv("NH_SEED");
    srand(seed_env ? (unsigned)strtoul(seed_env, NULL, 10) : (unsigned)time(NULL));

    float ep_score = 0, ep_len = 0, ep_depth = 0, ep_xp = 0, ep_gt = 0;
    float acts_f[DEMO_NUM_HEADS];
    for (long t = 0; t < max_steps; t++) {
        demo_step_once(net, &env, acts_f, &ep_score, &ep_len,
                       &ep_depth, &ep_xp, &ep_gt);
        if (frame_ms > 0) {
            c_render(&env);
            usleep(frame_ms * 1000);
        }
    }
    if (env.log.n > 0)
        printf("episodes=%.0f  avg_score=%.1f  avg_max_depth=%.2f  avg_xp=%.2f\n",
               env.log.n, env.log.score / env.log.n,
               env.log.max_depth / env.log.n, env.log.max_xp_level / env.log.n);
    env_close(&env);
    free_mingru(net->mingru);
    free(net->md); free(net->hidden); free(net->e_eff); free(net);
    free(w);
}

// ./nethack                  interactive TTY (space / shift+space)
// ./nethack N 0              headless N steps
// ./nethack N MS             auto-run N steps at MS ms/frame
// NH_WEIGHTS=... NH_SEED=...
int main(int argc, char** argv) {
    long max_steps = (argc >= 2) ? atol(argv[1]) : 1000000;
    int interactive = isatty(STDIN_FILENO);
    int frame_ms = 50;
    if (argc >= 3) {
        frame_ms = atoi(argv[2]);
        interactive = 0;   // explicit frame timing => auto mode
    } else if (!interactive) {
        frame_ms = 0;      // piped/non-TTY default: headless auto-run
    }
    if (interactive) run_demo_interactive(max_steps);
    else run_demo_auto(max_steps, frame_ms);
    return 0;
}
