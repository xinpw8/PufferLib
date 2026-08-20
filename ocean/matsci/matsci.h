#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <lammps/library.h>
#include "raylib.h"
typedef float obs_t;
#include "pufferenv.h"

#define MAX_AGENTS 256
#define ACT_SIZES {1, 1, 1}
#define NUM_ATNS 3
#define OBS_SIZE 3

#define WIDTH 1080
#define HEIGHT 720

const Color PUFF_RED = (Color){187, 0, 0, 255};
const Color PUFF_CYAN = (Color){0, 187, 187, 255};
const Color PUFF_WHITE = (Color){241, 241, 241, 241};
const Color PUFF_BACKGROUND = (Color){6, 24, 24, 255};

typedef struct {
    float x, y, z;
} Vec3;

static inline float clampf(float v, float min, float max) {
    if (v < min)
        return min;
    if (v > max)
        return max;
    return v;
}

static inline float rndf(float a, float b) {
    return a + ((float)rand() / (float)RAND_MAX) * (b - a);
}

static inline Vec3 add3(Vec3 a, Vec3 b) { return (Vec3){a.x + b.x, a.y + b.y, a.z + b.z}; }

static inline Vec3 sub3(Vec3 a, Vec3 b) { return (Vec3){a.x - b.x, a.y - b.y, a.z - b.z}; }

static inline Vec3 scalmul3(Vec3 a, float b) { return (Vec3){a.x * b, a.y * b, a.z * b}; }

static inline float dot3(Vec3 a, Vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

static inline float norm3(Vec3 a) { return sqrtf(dot3(a, a)); }

static inline void clamp3(Vec3 *vec, float min, float max) {
    vec->x = clampf(vec->x, min, max);
    vec->y = clampf(vec->y, min, max);
    vec->z = clampf(vec->z, min, max);
}


// Only use floats!
struct Log {
    float score;
    float n; // Required as the last field 
};

typedef struct {
    Camera3D camera;
    float width;
    float height;

    float camera_distance;
    float camera_azimuth;
    float camera_elevation;
    bool is_dragging;
    Vector2 last_mouse_pos;
} Client;

struct Env {
    Log log;
    Agent agents[MAX_AGENTS];
    int tag;
    int boundary_reached;
    int num_agents;
    Vec3 goal;
    int tick;
    Client* client;
    void* handle;
};
typedef Env Matsci;

void init(Matsci* env);

void puf_init(Env* env, Dict* kwargs) {
    env->num_agents = dict_get(kwargs, "num_agents");
    if (env->num_agents > MAX_AGENTS) {
        fprintf(stderr, "matsci: num_agents %d > MAX_AGENTS %d\n",
                env->num_agents, MAX_AGENTS);
        exit(1);
    }
    for (int i = 0; i < env->num_agents; i++) {
        env->agents[i].policy = 0;
        env->agents[i].action_mask = NULL;
    }
    init(env);
}

void puf_log(Log* log, Dict* out) {
    dict_set(out, "score", log->score);
    dict_set(out, "n", log->n);
}

void init(Matsci* env) {
  void *handle;
  const char *lmpargv[] = { "liblammps", "-log", "none", "-screen", "none"};
  int lmpargc = sizeof(lmpargv)/sizeof(const char *);

  /* create LAMMPS instance */
  handle = lammps_open_no_mpi(lmpargc, (char **)lmpargv, NULL);
  if (handle == NULL) {
    printf("LAMMPS initialization failed\n");
    lammps_mpi_finalize();
  }

  // Setup the basic simulation parameters via string commands
  lammps_command(handle, "units lj");
  lammps_command(handle, "dimension 3");
  lammps_command(handle, "boundary p p p");
  lammps_command(handle, "atom_style atomic");
  lammps_command(handle, "pair_style zero 1.0 nocoeff");  // Dummy pair style for no interactions
  lammps_command(handle, "region box block -10 10 -10 10 -10 10");
  lammps_command(handle, "create_box 1 box");
  lammps_command(handle, "mass 1 1.0");
  lammps_command(handle, "pair_coeff 1 1 1.0 1.0");

  lammps_command(handle, "region randbox block -10 10 -10 10 -10 10");
  char cmd[256];
  int seed = 123;
  snprintf(cmd, sizeof(cmd), "create_atoms 1 random %d %d randbox overlap 0.8", env->num_agents, seed);
  lammps_command(handle, cmd);

  // Setup for running simulations (timestep and integrator)
  lammps_command(handle, "timestep 0.5");
  lammps_command(handle, "fix 1 all nve");

  // Initialize
  lammps_command(handle, "run 0");
  env->handle = handle;
}

void compute_observations(Matsci* env) {
    double** x = (double **) lammps_extract_atom(env->handle, "x");
    for (int i=0; i<env->num_agents; i++) {
        obs_t* obs = env->agents[i].observations;
        obs[0] = x[i][0] - env->goal.x;
        obs[1] = x[i][1] - env->goal.y;
        obs[2] = x[i][2] - env->goal.z;
    }
}

void reset_atom(Matsci* env, double** x, int i) {
    x[i][0] = rndf(-10.0f, 10.0f);
    x[i][1] = rndf(-10.0f, 10.0f);
    x[i][2] = rndf(-10.0f, 10.0f);
}

void puf_reset(Matsci* env) {
    void* handle = env->handle;
    double** x = (double **) lammps_extract_atom(handle, "x");
    for (int i=0; i<env->num_agents; i++) {
	reset_atom(env, x, i);
    }
    env->goal.x = rndf(-10.0f, 10.0f);
    env->goal.y = rndf(-10.0f, 10.0f);
    env->goal.z = rndf(-10.0f, 10.0f);
    env->tick = 0;
}

void puf_step(Matsci* env) {
    void* handle = env->handle;
    env->tick++;

    if (env->tick >= 1024) {
        puf_reset(env);
        for (int i=0; i<env->num_agents; i++) {
            env->agents[i].rewards[0] = -1;
            env->agents[i].terminals[0] = 1;
            env->log.n += 1;
	}
	return;
    }

    double **v = (double **) lammps_extract_atom(handle, "v");
    for (int i=0; i<env->num_agents; i++) {
	env->agents[i].rewards[0] = 0;
	env->agents[i].terminals[0] = 0;

        v[i][0] = env->agents[i].actions[0];
        v[i][1] = env->agents[i].actions[1];
        v[i][2] = env->agents[i].actions[2];
    }

    lammps_command(handle, "run 1");

    double** x = (double **) lammps_extract_atom(handle, "x");
    for (int i=0; i<env->num_agents; i++) {
        Vec3 pos = (Vec3){x[i][0], x[i][1], x[i][2]};
        float dist = norm3(sub3(pos, env->goal));

        if (dist > 20.0f) {
            reset_atom(env, x, i);
            env->agents[i].rewards[0] = -1;
            env->agents[i].terminals[0] = 1;
            env->log.n += 1;
	}

        if (dist < 1.0f) {
           reset_atom(env, x, i);
           env->agents[i].rewards[0] = 1;
           env->agents[i].terminals[0] = 1;
           env->log.score += 1;
           env->log.n += 1;
	}
    }

    compute_observations(env);
}

static void update_camera_position(Client *c) {
    float r = c->camera_distance;
    float az = c->camera_azimuth;
    float el = c->camera_elevation;

    float x = r * cosf(el) * cosf(az);
    float y = r * cosf(el) * sinf(az);
    float z = r * sinf(el);

    c->camera.position = (Vector3){x, y, z};
    c->camera.target = (Vector3){0, 0, 0};
}

void handle_camera_controls(Client *client) {
    Vector2 mouse_pos = GetMousePosition();

    if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) {
        client->is_dragging = true;
        client->last_mouse_pos = mouse_pos;
    }

    if (IsMouseButtonReleased(MOUSE_BUTTON_LEFT)) {
        client->is_dragging = false;
    }

    if (client->is_dragging && IsMouseButtonDown(MOUSE_BUTTON_LEFT)) {
        Vector2 mouse_delta = {mouse_pos.x - client->last_mouse_pos.x,
                               mouse_pos.y - client->last_mouse_pos.y};

        float sensitivity = 0.005f;

        client->camera_azimuth -= mouse_delta.x * sensitivity;

        client->camera_elevation += mouse_delta.y * sensitivity;
        client->camera_elevation =
            clampf(client->camera_elevation, -PI / 2.0f + 0.1f, PI / 2.0f - 0.1f);

        client->last_mouse_pos = mouse_pos;

        update_camera_position(client);
    }

    float wheel = GetMouseWheelMove();
    if (wheel != 0) {
        client->camera_distance -= wheel * 2.0f;
        client->camera_distance = clampf(client->camera_distance, 5.0f, 50.0f);
        update_camera_position(client);
    }
}

void puf_close(Matsci* env) {
    /*
    if (IsWindowReady()) {
        CloseWindow();
    }
    */
}

void puf_render(Matsci* env) {
    if (!IsWindowReady()) {
        Client *client = (Client *)calloc(1, sizeof(Client));

        client->width = WIDTH;
        client->height = HEIGHT;

        SetConfigFlags(FLAG_MSAA_4X_HINT); // antialiasing
        InitWindow(WIDTH, HEIGHT, "PufferLib DroneSwarm");

        #ifndef __EMSCRIPTEN__
            SetTargetFPS(60);
        #endif

        client->camera_distance = 40.0f;
        client->camera_azimuth = 0.0f;
        client->camera_elevation = PI / 10.0f;
        client->is_dragging = false;
        client->last_mouse_pos = (Vector2){0.0f, 0.0f};

        client->camera.up = (Vector3){0.0f, 0.0f, 1.0f};
        client->camera.fovy = 45.0f;
        client->camera.projection = CAMERA_PERSPECTIVE;
	env->client = client;

        update_camera_position(client);
    }

    if (WindowShouldClose()) {
        puf_close(env);
        exit(0);
    }

    if (IsKeyDown(KEY_ESCAPE)) {

        puf_close(env);
        exit(0);
    }

    handle_camera_controls(env->client);

    Client *client = env->client;

    BeginDrawing();
    ClearBackground(PUFF_BACKGROUND);
    BeginMode3D(client->camera);
    DrawCubeWires((Vector3){0.0f, 0.0f, 0.0f}, 20.0f, 20.0f, 20.0f, WHITE);

    double** x = (double **) lammps_extract_atom(env->handle, "x");
    for (int i=0; i<env->num_agents; i++) {
        DrawSphere((Vector3){x[i][0], x[i][1], x[i][2]}, 0.1f, PUFF_CYAN);
    }

    DrawSphere((Vector3){env->goal.x, env->goal.y, env->goal.z}, 0.1f, PUFF_RED);
    EndMode3D();

    DrawText("Left click + drag: Rotate camera", 10, 10, 16, PUFF_WHITE);
    DrawText("Mouse wheel: Zoom in/out", 10, 30, 16, PUFF_WHITE);

    EndDrawing();
    puf_web_vsync();
}


