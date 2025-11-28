/*
 * Diablo (DevilutionX-AI) native PufferLib environment
 *
 * This interfaces with the DevilutionX game via shared memory.
 * The game binary runs as a separate process; this C code handles
 * observation computation and action submission via mmap.
 */

#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <linux/futex.h>
#include <sys/syscall.h>
#include <errno.h>

/* Dungeon dimensions */
#define DUNGEON_WIDTH 112
#define DUNGEON_HEIGHT 112

/* View parameters */
#define VIEW_RADIUS 10
#define VIEW_SIZE (2 * VIEW_RADIUS + 1)  /* 21x21 */

/* Observation channels: 19 env flags + 5 status */
#define ENV_FLAG_COUNT 19
#define ENV_STATUS_LEN 5
#define OBS_CHANNELS (ENV_FLAG_COUNT + ENV_STATUS_LEN)  /* 24 */

/* Ring buffer constants */
#define RING_QUEUE_CAPACITY 128
#define RING_QUEUE_MASK (RING_QUEUE_CAPACITY - 1)

/* Actions */
#define ACTION_STAND 0
#define ACTION_UP 1
#define ACTION_DOWN 2
#define ACTION_LEFT 3
#define ACTION_RIGHT 4
#define ACTION_UP_LEFT 5
#define ACTION_UP_RIGHT 6
#define ACTION_DOWN_LEFT 7
#define ACTION_DOWN_RIGHT 8
#define ACTION_PRIMARY 9
#define ACTION_SECONDARY 10

/* Ring entry key codes (from ring.py) */
#define RING_KEY_LEFT     (1 << 0)
#define RING_KEY_RIGHT    (1 << 1)
#define RING_KEY_UP       (1 << 2)
#define RING_KEY_DOWN     (1 << 3)
#define RING_KEY_X        (1 << 4)
#define RING_KEY_Y        (1 << 5)
#define RING_KEY_A        (1 << 6)
#define RING_KEY_B        (1 << 7)
#define RING_KEY_NEW      (1 << 8)
#define RING_KEY_SAVE     (1 << 9)
#define RING_KEY_LOAD     (1 << 10)
#define RING_KEY_PAUSE    (1 << 11)
#define RING_KEY_NOOP     (1 << 12)
#define RING_KEY_SET_GOAL (1 << 13)
#define RING_F_SINGLE_TICK (1U << 31)
#define RING_EVENT_STEP_FINISHED (1 << 30)

/* Environment flags (from diablo_state.py EnvironmentFlag) */
#define EF_PLAYER      (1 << 0)
#define EF_WALL        (1 << 1)
#define EF_PREV_TRIG   (1 << 2)
#define EF_NEXT_TRIG   (1 << 3)
#define EF_WARP_TRIG   (1 << 4)
#define EF_DOOR        (1 << 5)
#define EF_MISSILE     (1 << 6)
#define EF_MONSTER     (1 << 7)
#define EF_UNK_OBJ     (1 << 8)
#define EF_CRUCIFIX    (1 << 9)
#define EF_BARREL      (1 << 10)
#define EF_CHEST       (1 << 11)
#define EF_SARCOPHAG   (1 << 12)
#define EF_ITEM        (1 << 13)
#define EF_EXPLORED    (1 << 14)
#define EF_VISIBLE     (1 << 15)
#define EF_INTERACT    (1 << 16)
#define EF_OPEN        (1 << 17)
#define EF_GOAL        (1 << 18)

/* DungeonFlag bits (from devilutionx.py) */
#define DF_EXPLORED 0x80
#define DF_VISIBLE  0x01
#define DF_MISSILE  0x02

/* Object types for identification */
#define OBJ_BARREL    11
#define OBJ_BARRELEX  47
#define OBJ_POD       60
#define OBJ_PODEX     61
#define OBJ_URN       62
#define OBJ_URNEX     63
#define OBJ_CRUX1     1
#define OBJ_CRUX2     2
#define OBJ_CRUX3     3
#define OBJ_CHEST1    14
#define OBJ_CHEST2    15
#define OBJ_CHEST3    16
#define OBJ_TCHEST1   76
#define OBJ_TCHEST2   77
#define OBJ_TCHEST3   78
#define OBJ_SIGNCHEST 87
#define OBJ_SARC      25
#define OBJ_L5SARC    71

/* Trigger types */
#define TRIG_ENTRANCE  0
#define TRIG_EXIT      1
#define TRIG_WARP      2

/* Status normalization */
#define ENV_STATUS_HIGH 0xFFFFF

/* Memory offsets from devilutionx.py VARS (relative to base) */
typedef struct {
    uint32_t input_queue;
    uint32_t events_queue;
    uint32_t player;
    uint32_t game_ticks;
    uint32_t active_monster_count;
    uint32_t monsters;
    uint32_t objects;
    uint32_t active_objects;
    uint32_t dItem;
    uint32_t dFlags;
    uint32_t dMonster;
    uint32_t dObject;
    uint32_t trigs;
    uint32_t numtrigs;
    uint32_t sol_data;
} MemoryOffsets;

/* Ring buffer entry */
typedef struct {
    uint32_t en_type;
    uint32_t en_tag;
    uint32_t en_data1;
    uint32_t en_data2;
} RingEntry;

/* Ring queue */
typedef struct {
    uint32_t write_idx;
    uint32_t read_idx;
    RingEntry array[RING_QUEUE_CAPACITY];
} RingQueue;

/* Trigger structure (simplified) */
typedef struct {
    int32_t position_x;
    int32_t position_y;
    int32_t _tmsg;
    int32_t _tlvl;
} TriggerStruct;

/* Object structure (simplified - key fields only) */
typedef struct {
    uint8_t _padding1[4];  /* position Point<int> = 8 bytes */
    int32_t _otype;
    uint8_t _padding2[44]; /* skip to selectionRegion at offset 52 */
    uint8_t selectionRegion;
    uint8_t _padding3[3];
    uint8_t _oBreak;
    uint8_t _padding4[7];
    uint8_t _oDoorFlag;
    uint8_t _padding5[3];
    int32_t _oVar4;
} __attribute__((packed)) ObjectPartial;

/* Required struct for logging */
typedef struct {
    float perf;
    float score;
    float episode_return;
    float episode_length;
    float success_rate;
    float n;
} Log;

typedef struct Diablo Diablo;

struct Diablo {
    Log log;

    /* Pointers to numpy arrays (from PufferLib) */
    float* observations;      /* (VIEW_SIZE, VIEW_SIZE, OBS_CHANNELS) float32 */
    int* actions;             /* Discrete action */
    float* rewards;
    unsigned char* terminals;

    /* Shared memory */
    void* mmap_base;
    size_t mmap_size;
    int mmap_fd;
    uint32_t base_offset;

    /* Memory offsets (absolute from mmap_base) */
    MemoryOffsets offsets;

    /* Episode tracking */
    float ep_return;
    int ep_len;
    int max_steps;
    int view_radius;

    /* Game state cache */
    int32_t goal_x;
    int32_t goal_y;
    int game_ticks_per_step;

    /* Flags */
    int initialized;
    int step_mode;
};

/* Futex helpers */
static inline int futex_wait(uint32_t* addr, uint32_t expected) {
    return syscall(SYS_futex, addr, FUTEX_WAIT, expected, NULL, NULL, 0);
}

static inline int futex_wake(uint32_t* addr) {
    return syscall(SYS_futex, addr, FUTEX_WAKE, 1, NULL, NULL, 0);
}

/* Ring buffer operations */
static inline int ring_has_capacity(RingQueue* ring) {
    return (ring->write_idx - ring->read_idx) < RING_QUEUE_CAPACITY;
}

static inline RingEntry* ring_get_submit_entry(RingQueue* ring) {
    return &ring->array[ring->write_idx & RING_QUEUE_MASK];
}

static inline void ring_submit(RingQueue* ring) {
    ring->write_idx++;
    futex_wake(&ring->write_idx);
}

static inline RingEntry* ring_get_retrieve_entry(RingQueue* ring) {
    if (ring->write_idx == ring->read_idx) {
        return NULL;
    }
    return &ring->array[ring->read_idx & RING_QUEUE_MASK];
}

static inline void ring_retrieve(RingQueue* ring) {
    ring->read_idx++;
    futex_wake(&ring->read_idx);
}

static inline void ring_wait_submitted(RingQueue* ring) {
    uint32_t read_idx = ring->read_idx;
    while (ring->write_idx == read_idx) {
        futex_wait(&ring->write_idx, read_idx);
    }
}

/* Memory access helpers */
static inline void* mem_ptr(Diablo* env, uint32_t offset) {
    return (char*)env->mmap_base + offset;
}

static inline RingQueue* get_input_queue(Diablo* env) {
    return (RingQueue*)mem_ptr(env, env->offsets.input_queue);
}

static inline RingQueue* get_events_queue(Diablo* env) {
    return (RingQueue*)mem_ptr(env, env->offsets.events_queue);
}

static inline int32_t get_player_x(Diablo* env) {
    /* Player position.x is at offset 0 in Player struct */
    int32_t* pos = (int32_t*)mem_ptr(env, env->offsets.player);
    return pos[0];
}

static inline int32_t get_player_y(Diablo* env) {
    int32_t* pos = (int32_t*)mem_ptr(env, env->offsets.player);
    return pos[1];
}

static inline int32_t get_player_hp(Diablo* env) {
    /* _pHitPoints is at a specific offset in Player - needs verification */
    /* For now, return a placeholder */
    return 100;
}

static inline int32_t get_player_mode(Diablo* env) {
    /* _pmode field offset - needs verification */
    return 0;
}

static inline uint8_t get_dFlags(Diablo* env, int x, int y) {
    uint8_t* flags = (uint8_t*)mem_ptr(env, env->offsets.dFlags);
    return flags[y * DUNGEON_WIDTH + x];
}

static inline int16_t get_dMonster(Diablo* env, int x, int y) {
    int16_t* mons = (int16_t*)mem_ptr(env, env->offsets.dMonster);
    return mons[y * DUNGEON_WIDTH + x];
}

static inline int8_t get_dObject(Diablo* env, int x, int y) {
    int8_t* objs = (int8_t*)mem_ptr(env, env->offsets.dObject);
    return objs[y * DUNGEON_WIDTH + x];
}

static inline int8_t get_dItem(Diablo* env, int x, int y) {
    int8_t* items = (int8_t*)mem_ptr(env, env->offsets.dItem);
    return items[y * DUNGEON_WIDTH + x];
}

static inline int get_num_trigs(Diablo* env) {
    int32_t* num = (int32_t*)mem_ptr(env, env->offsets.numtrigs);
    return *num;
}

static inline TriggerStruct* get_trigger(Diablo* env, int idx) {
    TriggerStruct* trigs = (TriggerStruct*)mem_ptr(env, env->offsets.trigs);
    return &trigs[idx];
}

static inline size_t get_active_monster_count(Diablo* env) {
    size_t* cnt = (size_t*)mem_ptr(env, env->offsets.active_monster_count);
    return *cnt;
}

/* Check if tile is a wall using SOLData */
static inline int is_wall(Diablo* env, int x, int y) {
    /* Simplified: check if dFlags has solid bit or check SOLData */
    /* For now, approximate using dFlags */
    uint8_t flags = get_dFlags(env, x, y);
    /* A tile is walkable if explored and not blocked */
    /* This is a simplification - real implementation needs SOLData */
    return 0;  /* TODO: implement properly with SOLData */
}

/* Submit an action via ring buffer */
static void submit_action(Diablo* env, int action) {
    RingQueue* input = get_input_queue(env);

    if (!ring_has_capacity(input)) {
        return;  /* Queue full, skip */
    }

    RingEntry* entry = ring_get_submit_entry(input);
    entry->en_tag = 0;
    entry->en_data1 = 0;
    entry->en_data2 = 0;

    /* Convert action to key code */
    uint32_t key = 0;
    switch (action) {
        case ACTION_STAND:
            key = RING_KEY_NOOP;
            break;
        case ACTION_UP:
            key = RING_KEY_UP;
            break;
        case ACTION_DOWN:
            key = RING_KEY_DOWN;
            break;
        case ACTION_LEFT:
            key = RING_KEY_LEFT;
            break;
        case ACTION_RIGHT:
            key = RING_KEY_RIGHT;
            break;
        case ACTION_UP_LEFT:
            key = RING_KEY_UP | RING_KEY_LEFT;
            break;
        case ACTION_UP_RIGHT:
            key = RING_KEY_UP | RING_KEY_RIGHT;
            break;
        case ACTION_DOWN_LEFT:
            key = RING_KEY_DOWN | RING_KEY_LEFT;
            break;
        case ACTION_DOWN_RIGHT:
            key = RING_KEY_DOWN | RING_KEY_RIGHT;
            break;
        case ACTION_PRIMARY:
            key = RING_KEY_A;
            break;
        case ACTION_SECONDARY:
            key = RING_KEY_B;
            break;
        default:
            key = RING_KEY_NOOP;
    }

    /* Always use single tick press (matches original Python env behavior) */
    entry->en_type = key | RING_F_SINGLE_TICK;
    ring_submit(input);
}

/* Wait for step to complete (events queue feedback) */
static void wait_step_complete(Diablo* env) {
    if (!env->step_mode) return;

    RingQueue* events = get_events_queue(env);
    ring_wait_submitted(events);

    RingEntry* entry = ring_get_retrieve_entry(events);
    if (entry && (entry->en_type & RING_EVENT_STEP_FINISHED)) {
        ring_retrieve(events);
    }
}

/* Compute 21x21 observation centered on player */
static void compute_observation(Diablo* env) {
    int px = get_player_x(env);
    int py = get_player_y(env);
    int radius = env->view_radius;
    int size = 2 * radius + 1;

    /* Clear observations */
    memset(env->observations, 0, size * size * OBS_CHANNELS * sizeof(float));

    /* Compute env status (global values broadcast to all cells) */
    float status[ENV_STATUS_LEN];
    status[0] = (float)get_active_monster_count(env) / ENV_STATUS_HIGH;
    status[1] = (float)get_player_hp(env) / ENV_STATUS_HIGH;
    status[2] = (float)get_player_mode(env) / ENV_STATUS_HIGH;
    status[3] = (float)px / ENV_STATUS_HIGH;
    status[4] = (float)py / ENV_STATUS_HIGH;

    /* Clamp status values */
    for (int i = 0; i < ENV_STATUS_LEN; i++) {
        if (status[i] < 0.0f) status[i] = 0.0f;
        if (status[i] > 1.0f) status[i] = 1.0f;
    }

    /* Iterate over view window */
    for (int j = 0; j < size; j++) {
        for (int i = 0; i < size; i++) {
            int wx = px - radius + i;
            int wy = py - radius + j;

            /* Skip out of bounds */
            if (wx < 0 || wx >= DUNGEON_WIDTH || wy < 0 || wy >= DUNGEON_HEIGHT) {
                continue;
            }

            uint32_t tile_flags = 0;
            uint8_t dflags = get_dFlags(env, wx, wy);

            /* Explored/Visible */
            if (dflags & DF_EXPLORED) tile_flags |= EF_EXPLORED;
            if (dflags & DF_VISIBLE) tile_flags |= EF_VISIBLE;

            /* Only show details if explored */
            if (tile_flags & EF_EXPLORED) {
                /* Wall detection - simplified */
                /* TODO: proper SOLData check */

                /* Triggers */
                int num_trigs = get_num_trigs(env);
                for (int t = 0; t < num_trigs; t++) {
                    TriggerStruct* trig = get_trigger(env, t);
                    if (trig->position_x == wx && trig->position_y == wy) {
                        /* Determine trigger type */
                        if (trig->_tlvl > 0) {
                            tile_flags |= EF_NEXT_TRIG;
                        } else if (trig->_tlvl < 0) {
                            tile_flags |= EF_PREV_TRIG;
                        } else {
                            tile_flags |= EF_WARP_TRIG;
                        }
                    }
                }

                /* Objects (doors handled here) */
                int8_t obj_id = get_dObject(env, wx, wy);
                if (obj_id != 0) {
                    /* Would need to check object type for door */
                    /* Simplified: mark as door if object present */
                    tile_flags |= EF_DOOR;
                }
            }

            /* Only show dynamic content if visible */
            if (tile_flags & EF_VISIBLE) {
                /* Goal */
                if (wx == env->goal_x && wy == env->goal_y) {
                    tile_flags |= EF_GOAL;
                }

                /* Missiles */
                if (dflags & DF_MISSILE) {
                    tile_flags |= EF_MISSILE;
                }

                /* Monsters */
                int16_t mon = get_dMonster(env, wx, wy);
                if (mon > 0) {
                    tile_flags |= EF_MONSTER;
                }

                /* Items */
                int8_t item = get_dItem(env, wx, wy);
                if (item > 0) {
                    tile_flags |= EF_ITEM;
                }

                /* Objects (barrels, chests, etc.) */
                int8_t obj_id = get_dObject(env, wx, wy);
                if (obj_id != 0) {
                    /* Simplified object type detection */
                    /* TODO: proper object type checking */
                    tile_flags |= EF_BARREL;  /* placeholder */
                }
            }

            /* Player */
            if (wx == px && wy == py) {
                tile_flags |= EF_PLAYER;
            }

            /* Write observation: convert bitfield to float channels */
            int obs_idx = (j * size + i) * OBS_CHANNELS;

            /* Environment flag channels (first 19) */
            for (int c = 0; c < ENV_FLAG_COUNT; c++) {
                env->observations[obs_idx + c] = ((tile_flags >> c) & 1) ? 1.0f : 0.0f;
            }

            /* Status channels (last 5) - broadcast same values */
            for (int c = 0; c < ENV_STATUS_LEN; c++) {
                env->observations[obs_idx + ENV_FLAG_COUNT + c] = status[c];
            }
        }
    }
}

/* Add episode stats to log */
void add_log(Diablo* env) {
    float success = (env->rewards[0] > 0) ? 1.0f : 0.0f;
    env->log.perf += success;
    env->log.score += env->ep_return;
    env->log.episode_return += env->ep_return;
    env->log.episode_length += env->ep_len;
    env->log.success_rate += success;
    env->log.n += 1.0f;
}

/* Reset environment */
void c_reset(Diablo* env) {
    if (!env->initialized) return;

    /* NOTE: Don't send new game command on reset - the game is already initialized.
     * The Python side handles game launching. We just need to reset our tracking
     * and compute the initial observation from the current game state.
     *
     * If we do need to reset the game (e.g., after terminal), the step function
     * handles that by auto-resetting.
     */

    /* Reset episode tracking */
    env->ep_return = 0.0f;
    env->ep_len = 0;

    /* Compute initial observation from current game state */
    compute_observation(env);
}

/* Step environment */
void c_step(Diablo* env) {
    if (!env->initialized) {
        env->terminals[0] = 0;
        env->rewards[0] = 0;
        return;
    }

    /* Submit action */
    int action = env->actions[0];
    submit_action(env, action);

    /* Small delay to let game process the action
     * TODO: Implement proper synchronization via events queue
     */
    usleep(10000);  /* 10ms delay */

    env->ep_len++;

    /* Compute observation */
    compute_observation(env);

    /* Check termination conditions */
    int terminated = 0;
    float reward = -0.001f;  /* Small step penalty to encourage efficiency */

    /* Check if reached goal (only if goal is set, i.e., not at 0,0) */
    int px = get_player_x(env);
    int py = get_player_y(env);
    if ((env->goal_x != 0 || env->goal_y != 0) &&
        px == env->goal_x && py == env->goal_y) {
        reward = 1.0f;
        terminated = 1;
    }

    /* Check max steps */
    if (env->ep_len >= env->max_steps) {
        terminated = 1;
    }

    /* TODO: check player death, find level exit trigger, etc. */

    env->rewards[0] = reward;
    env->ep_return += reward;
    env->terminals[0] = terminated ? 1 : 0;

    if (terminated) {
        add_log(env);
        c_reset(env);
    }
}

/* Render (no-op for now, game handles its own rendering) */
void c_render(Diablo* env) {
    /* DevilutionX handles rendering */
}

/* Close environment */
void c_close(Diablo* env) {
    if (env->mmap_base && env->mmap_base != MAP_FAILED) {
        munmap(env->mmap_base, env->mmap_size);
        env->mmap_base = NULL;
    }
    if (env->mmap_fd >= 0) {
        close(env->mmap_fd);
        env->mmap_fd = -1;
    }
    env->initialized = 0;
}

/* Initialize shared memory mapping */
int init_mmap(Diablo* env, const char* mmap_path, uint32_t base_offset) {
    env->mmap_fd = open(mmap_path, O_RDWR);
    if (env->mmap_fd < 0) {
        return -1;
    }

    struct stat st;
    if (fstat(env->mmap_fd, &st) < 0) {
        close(env->mmap_fd);
        env->mmap_fd = -1;
        return -1;
    }

    env->mmap_size = st.st_size;
    env->mmap_base = mmap(NULL, env->mmap_size, PROT_READ | PROT_WRITE,
                          MAP_SHARED, env->mmap_fd, 0);

    if (env->mmap_base == MAP_FAILED) {
        close(env->mmap_fd);
        env->mmap_fd = -1;
        return -1;
    }

    env->base_offset = base_offset;

    /* Set up memory offsets (from devilutionx.py VARS) */
    /* These are absolute addresses; subtract base_offset to get file offset */
    env->offsets.input_queue = 6292480 - base_offset;
    env->offsets.events_queue = 6294592 - base_offset;
    env->offsets.player = 6274528 - base_offset;
    env->offsets.game_ticks = 6294544 - base_offset;
    env->offsets.active_monster_count = 6188672 - base_offset;
    env->offsets.monsters = 5879616 - base_offset;
    env->offsets.objects = 5900448 - base_offset;
    env->offsets.active_objects = 6122464 - base_offset;
    env->offsets.dItem = 5954816 - base_offset;
    env->offsets.dFlags = 6090400 - base_offset;
    env->offsets.dMonster = 5924224 - base_offset;
    env->offsets.dObject = 5911648 - base_offset;
    env->offsets.trigs = 6121792 - base_offset;
    env->offsets.numtrigs = 6121776 - base_offset;
    env->offsets.sol_data = 6022272 - base_offset;

    env->initialized = 1;
    return 0;
}
