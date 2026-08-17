#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <assert.h>

#include "raylib.h"
#include "level_generation/puzzle_types.h"
typedef unsigned char obs_t;
#include "pufferenv.h"

#define BOARD_IDX(cols, r, c) ((r) * (cols) + (c))
#define LASER_PUZZLE_LEVELS_PATH "resources/laser_puzzle/laser_puzzle_levels.bin"

// observations: 6*6 board, one byte per cell:
// 0 empty, 1-8 laser ids 0-7, 9-16 sensor ids 0-7, 17 mirror /, 18 mirror \'
#define LASER_PUZZLE_OBS_SIZE (INIT_ROWS * INIT_COLS)
#define OBS_EMPTY 0
#define OBS_LASER 1
#define OBS_SENSOR (OBS_LASER + MAX_LASERS)
#define OBS_MIRROR_RIGHT (OBS_SENSOR + MAX_LASERS)
#define OBS_MIRROR_LEFT (OBS_MIRROR_RIGHT + 1)

// actions: 4 * 4 * 3, set mirror to none, left or right for each interior cell. discrete actions
#define ACTIONS_PER_CELL 3
#define INNER_ROWS (INIT_ROWS - 2)
#define INNER_COLS (INIT_COLS - 2)
#define NUM_ACTIONS (ACTIONS_PER_CELL * INNER_ROWS * INNER_COLS)

#define ACT_SIZES {NUM_ACTIONS}
#define OBS_SIZE (INIT_ROWS * INIT_COLS)
#define NUM_ATNS 1
#define PUF_STEPS_PER_SEC 3

static const int CELL_SIZE = 80;
static const Color LASER_COLORS[] = {SKYBLUE, RED, GREEN, YELLOW, BLUE, ORANGE, PURPLE, MAGENTA};

// Required struct. Only use floats!
struct Log {
    float perf; // Recommended 0-1 normalized single real number perf metric
    float score; // Recommended unnormalized single real number perf metric
    float episode_return; // Recommended metric: sum of agent rewards over episode
    float episode_length; // Recommended metric: number of steps of agent episode
    // Any extra fields you add here may be exported in binding.c
    float n; // Required as the last field
};

typedef struct {
    Texture2D sprites;
    Texture2D background;
    Font font;
    int assets_loaded;
} Client;

typedef struct {
    int optimal_mirrors;
    int sensor_count;
    Cell puzzle[INIT_ROWS][INIT_COLS];
} LaserPuzzleLevel;

struct Env {
    Log log;  // only stores results for completed episodes
    Agent agents[1];
    int tag;
    int boundary_reached;
    Client* client;

    // vecenv uses num_agents and rng; owns_buffers prevents freeing vecenv-owned buffers.
    int num_agents;
    unsigned int rng;
    int owns_buffers;

    int episode_length;
    int max_steps;  // max actions allowed before the episode is over
    float episode_return; // return for this episode

    // env specific
    int ROWS;
    int COLS;
    Cell *board;
    int sinks_found;
    int mirrors_placed;
    int moves_made;
    int total_sinks;
    int sink_hit_before[MAX_LASERS];
    int optimal_mirrors;
    int num_levels;
    LaserPuzzleLevel* levels;
};
typedef Env LaserPuzzle;

void load_laser_puzzle_levels(LaserPuzzle* env, const char* path) {
    FILE* file = fopen(path, "rb");
    assert(file);

    uint32_t header[3] = {0};
    assert(fread(header, sizeof(uint32_t), 3, file) == 3);

    int level_count = (int)header[2];
    assert(level_count > 0);
    LaserPuzzleLevel* levels = (LaserPuzzleLevel*)calloc((size_t)level_count, sizeof(LaserPuzzleLevel));

    for (int i = 0; i < level_count; i++) {
        fread(&levels[i].optimal_mirrors, sizeof(int), 1, file);
        fread(&levels[i].sensor_count, sizeof(int), 1, file);
        for (int r = 0; r < INIT_ROWS; r++) {
            for (int c = 0; c < INIT_COLS; c++) {
                uint8_t raw[4] = {0};
                fread(raw, sizeof(raw), 1, file);
                levels[i].puzzle[r][c] = (Cell){
                    .type = (CellType)raw[0],
                    .mirror = (MirrorState)raw[1],
                    .id = (int8_t)raw[2],
                };
            }
        }
    }

    fclose(file);
    env->levels = levels;
    env->num_levels = level_count;
}

Client* make_client() {
    Client* client = (Client*)calloc(1, sizeof(Client));
    InitWindow(800, 700, "laser puzzle");
    SetTargetFPS(60);

    client->sprites = LoadTexture("resources/shared/puffers.png");
    client->font = LoadFontEx("resources/shared/JetBrainsMono-SemiBold.ttf", 32, NULL, 0);
    client->assets_loaded = 1;
    return client;
}

void close_client(Client* client) {
    UnloadTexture(client->sprites);
    UnloadTexture(client->background);
    UnloadFont(client->font);
    client->assets_loaded = 0;
    if (IsWindowReady()) {
        CloseWindow();
    }
    free(client);
}

void puf_close(LaserPuzzle* env) {
    if (env->client != NULL) {
        close_client(env->client);
    }
    free(env->board);
    free(env->levels);
}

void add_log(LaserPuzzle* env) {
    float perf = 0.0f; // takes into account sinks + mirros placed, normalized
    if (env->mirrors_placed > 0) {
        perf = ((float)env->sinks_found * (float)env->optimal_mirrors)
            / ((float)env->total_sinks * (float)env->mirrors_placed);
    }

    float score = 0.0f; // takes into account sinks + mirros placed, unnormalized
    if (env->mirrors_placed > 0) {
        score = (float)env->sinks_found
            * ((float)env->optimal_mirrors / (float)env->mirrors_placed);
    }

    env->log.perf += perf;
    env->log.score += score;
    env->log.episode_return += env->episode_return;
    env->log.episode_length += env->episode_length;
    env->log.n += 1.0f;
}

void apply_action(LaserPuzzle* env) {
    int action = (int)env->agents[0].actions[0];

    int cell_idx = action / ACTIONS_PER_CELL;      // 0..15 (for a 6x6 grid)
    int mirror_action = action % ACTIONS_PER_CELL; // 0..2

    int r = cell_idx / INNER_COLS;                 // 0..3 interior row
    int c = cell_idx % INNER_COLS;                 // 0..3 interior col

    // +1 to skip the borders, since the actions only correspond to the inner rows
    Cell* cell = &env->board[BOARD_IDX(env->COLS, r + 1, c + 1)];
    cell->mirror = (MirrorState)mirror_action;
}

void compute_observations(LaserPuzzle* env) {
    obs_t* obs_buf = env->agents[0].observations;
    for (int r = 0; r < env->ROWS; r++) {
        for (int c = 0; c < env->COLS; c++) {
            Cell cell = env->board[BOARD_IDX(env->COLS, r, c)];
            unsigned char obs = OBS_EMPTY;

            if (cell.type == LASER) {
                obs = OBS_LASER + (unsigned char)cell.id;
            } else if (cell.type == SENSOR) {
                obs = OBS_SENSOR + (unsigned char)cell.id;
            } else if (cell.mirror == MIRROR_RIGHT) {
                obs = OBS_MIRROR_RIGHT;
            } else if (cell.mirror == MIRROR_LEFT) {
                obs = OBS_MIRROR_LEFT;
            }

            obs_buf[BOARD_IDX(env->COLS, r, c)] = obs;
        }
    }
}

// reset the env state (ignore rewards, terminals --> handled by puf_step)
void puf_reset(LaserPuzzle* env) {
    env->sinks_found = 0;
    env->mirrors_placed = 0;
    env->moves_made = 0;
    env->episode_length = 0;
    env->episode_return = 0.0f;

    memset(env->sink_hit_before, 0, sizeof(env->sink_hit_before));

    int level_index = rand_r(&env->rng) % env->num_levels;
    const LaserPuzzleLevel* level = &env->levels[level_index];
    env->total_sinks = level->sensor_count;
    env->optimal_mirrors = level->optimal_mirrors;

    memcpy(env->board, level->puzzle, sizeof(level->puzzle));

    compute_observations(env);
}

// Hold Left Shift + click a cell to cycle its mirror.
static int laser_puzzle_human_controls(LaserPuzzle *env) {
    if (!IsWindowReady() || !IsKeyDown(KEY_LEFT_SHIFT)) {
        return 0;
    }
    if (!IsMouseButtonPressed(MOUSE_LEFT_BUTTON)) {
        return -1;
    }
    int gridWidth = env->COLS * CELL_SIZE;
    int gridHeight = env->ROWS * CELL_SIZE;
    int offsetX = (GetScreenWidth() - gridWidth) / 2;
    int offsetY = (GetScreenHeight() - gridHeight) / 2;
    Vector2 mouse = GetMousePosition();
    int c = ((int)mouse.x - offsetX) / CELL_SIZE;
    int r = ((int)mouse.y - offsetY) / CELL_SIZE;
    if (r >= 1 && r < env->ROWS - 1 && c >= 1 && c < env->COLS - 1) {
        Cell* cell = &env->board[BOARD_IDX(env->COLS, r, c)];
        int mirror_action = (cell->mirror + 1) % ACTIONS_PER_CELL;
        int cell_idx = (r - 1) * INNER_COLS + (c - 1);
        env->agents[0].actions[0] = (float)(cell_idx * ACTIONS_PER_CELL + mirror_action);
        return 1;
    }
    return -1;
}

// advance state
void puf_step(LaserPuzzle* env) {
    if (laser_puzzle_human_controls(env) < 0) {
        return;
    }

    apply_action(env);
    env->moves_made++;

    // now we need to detect and update how many lasers are in thier sink and mirros are placed
    env->sinks_found = 0;
    env->mirrors_placed = 0;
    int new_sinks_hit = 0;
    for (int r = 0; r < env->ROWS; r++) {
        for (int c = 0; c < env->COLS; c++) {
            Cell boardCell = env->board[BOARD_IDX(env->COLS, r, c)];
            if (boardCell.mirror != MIRROR_NONE) {
                env->mirrors_placed++;
            }

            if (boardCell.type != LASER) {
                continue;
            }

            int laserId = boardCell.id;
            int curR = r;
            int curC = c;
            int dr = 0;
            int dc = 0;

            if (curR == 0) {
                dr = 1;
            } else if (curR == env->ROWS - 1) {
                dr = -1;
            } else if (curC == 0) {
                dc = 1;
            } else if (curC == env->COLS - 1) {
                dc = -1;
            }

            while (curR + dr >= 0 && curR + dr < env->ROWS && curC + dc >= 0 && curC + dc < env->COLS) {
                curR += dr;
                curC += dc;

                Cell hitCell = env->board[BOARD_IDX(env->COLS, curR, curC)];
                if (hitCell.type == SENSOR && hitCell.id == laserId) {
                    env->sinks_found++;

                    if (!env->sink_hit_before[laserId]) {
                        env->sink_hit_before[laserId] = 1;
                        new_sinks_hit++;
                    }
                } else if (hitCell.mirror == MIRROR_LEFT) {
                    int oldDr = dr;
                    dr = dc;
                    dc = oldDr;
                } else if (hitCell.mirror == MIRROR_RIGHT) {
                    int oldDr = dr;
                    dr = -dc;
                    dc = -oldDr;
                }
            }
        }
    }

    // handle the rewards, episode_length, terminal, episode_return
    // rewards: +1 for ending the episode optimally (minimal mirrors), +0.6 for ending the episode suboptimally, -0.01 per move, +0.3 for first time laser hit
    env->episode_length++;
    env->agents[0].rewards[0] = 0.3f * (float)new_sinks_hit;
    env->agents[0].terminals[0] = 0.0f;

    if (env->sinks_found == env->total_sinks) {
        env->agents[0].terminals[0] = 1.0f;
        if (env->mirrors_placed == env->optimal_mirrors) {
            env->agents[0].rewards[0] += 1.0f;
        } else {
            env->agents[0].rewards[0] += 0.6f;
        }
    } else if (env->episode_length >= env->max_steps) {
        env->agents[0].terminals[0] = 1.0f;
    }

    env->episode_return += env->agents[0].rewards[0];

    if (env->agents[0].terminals[0]) {
        add_log(env);
        puf_reset(env);
    }

    compute_observations(env);
}

void trace_laser(LaserPuzzle * env, int r, int c) {
    Cell laser = env->board[BOARD_IDX(env->COLS, r, c)];
    Color laserColor = LASER_COLORS[laser.id % 8];

    int dr = 0;
    int dc = 0;
    if (r == 0) {
        dr = 1;
    } else if (r == env->ROWS - 1) {
        dr = -1;
    } else if (c == 0) {
        dc = 1;
    } else if (c == env->COLS - 1) {
        dc = -1;
    }

    int gridWidth = env->COLS * CELL_SIZE;
    int gridHeight = env->ROWS * CELL_SIZE;
    int offsetX = (GetScreenWidth() - gridWidth) / 2;
    int offsetY = (GetScreenHeight() - gridHeight) / 2;

    int curR = r;
    int curC = c;

    while (curR + dr >= 0 && curR + dr < env->ROWS && curC + dc >= 0 && curC + dc < env->COLS) {
        int nextR = curR + dr;
        int nextC = curC + dc;

        Vector2 start = {
            offsetX + curC * CELL_SIZE + CELL_SIZE / 2.0f,
            offsetY + curR * CELL_SIZE + CELL_SIZE / 2.0f
        };
        Vector2 end = {
            offsetX + nextC * CELL_SIZE + CELL_SIZE / 2.0f,
            offsetY + nextR * CELL_SIZE + CELL_SIZE / 2.0f
        };

        // offset so that the puffer fish mouth not blocked by lasers
        if (env->board[BOARD_IDX(env->COLS, curR, curC)].type == LASER) {
            start.x += dc * 27.0f;
            start.y += dr * 27.0f;
        }

        DrawLineEx(start, end, 7, Fade(laserColor, 0.65f));
        DrawLineEx(start, end, 3, Fade(WHITE, 0.75f));

        // update current cell
        curR = nextR;
        curC = nextC;
        
        // update direction
        Cell cell = env->board[BOARD_IDX(env->COLS, curR, curC)];
        if (cell.mirror == MIRROR_LEFT) {
            int oldDr = dr;
            dr = dc;
            dc = oldDr;
        } else if (cell.mirror == MIRROR_RIGHT) {
            int oldDr = dr;
            dr = -dc;
            dc = -oldDr;
        }
    }
}

void draw_lasers(LaserPuzzle *env) {
    for (int r = 0; r < env->ROWS; r++) {
        for (int c = 0; c < env->COLS; c++) {
            if (env->board[BOARD_IDX(env->COLS, r, c)].type == LASER) {
                trace_laser(env, r, c);
            }
        }
    }
}

void puf_render(LaserPuzzle* env) {
    // this client loading here and "escape key to shutdown" is Puffer convention, needs to be like this for
    // puffer eval to work.
    if (env->client == NULL) {
        env->client = make_client();
    }
    Client* client = env->client;

    // Standard across our envs so exiting is always the same
    if (IsKeyDown(KEY_ESCAPE)) {
        exit(0);
    }

    laser_puzzle_human_controls(env);

    BeginDrawing();

    ClearBackground((Color){6, 24, 24, 255});

    // draw the centered grid
    int gridWidth = env->COLS * CELL_SIZE;
    int gridHeight = env->ROWS * CELL_SIZE;
    int offsetX = (GetScreenWidth() - gridWidth) / 2;
    int offsetY = (GetScreenHeight() - gridHeight) / 2;

    // Fixed layers (back → front). Do not mix types in one grid walk or
    // later cells paint over earlier ones.
    for (int r = 1; r < env->ROWS - 1; r++) {
        for (int c = 1; c < env->COLS - 1; c++) {
            int x = offsetX + c * CELL_SIZE;
            int y = offsetY + r * CELL_SIZE;
            DrawLineEx((Vector2){x + 20, y + 20}, (Vector2){x + CELL_SIZE - 20, y + CELL_SIZE - 20}, 2, Fade(LIGHTGRAY, 0.55f));
            DrawLineEx((Vector2){x + CELL_SIZE - 20, y + 20}, (Vector2){x + 20, y + CELL_SIZE - 20}, 2, Fade(LIGHTGRAY, 0.55f));
        }
    }

    draw_lasers(env);

    for (int r = 0; r < env->ROWS; r++) {
        for (int c = 0; c < env->COLS; c++) {
            Cell cell = env->board[BOARD_IDX(env->COLS, r, c)];
            if (cell.mirror == MIRROR_NONE) {
                continue;
            }
            int x = offsetX + c * CELL_SIZE;
            int y = offsetY + r * CELL_SIZE;
            if (cell.mirror == MIRROR_LEFT) {
                DrawLineEx((Vector2){x + 10, y + 10}, (Vector2){x + CELL_SIZE - 10, y + CELL_SIZE - 10}, 12, Fade(VIOLET, 0.55f));
                DrawLineEx((Vector2){x + 10, y + 10}, (Vector2){x + CELL_SIZE - 10, y + CELL_SIZE - 10}, 8, Fade(SKYBLUE, 0.9f));
                DrawLineEx((Vector2){x + 10, y + 10}, (Vector2){x + CELL_SIZE - 10, y + CELL_SIZE - 10}, 4, BLACK);
            } else {
                DrawLineEx((Vector2){x + CELL_SIZE - 10, y + 10}, (Vector2){x + 10, y + CELL_SIZE - 10}, 12, Fade(VIOLET, 0.55f));
                DrawLineEx((Vector2){x + CELL_SIZE - 10, y + 10}, (Vector2){x + 10, y + CELL_SIZE - 10}, 8, Fade(SKYBLUE, 0.9f));
                DrawLineEx((Vector2){x + CELL_SIZE - 10, y + 10}, (Vector2){x + 10, y + CELL_SIZE - 10}, 4, BLACK);
            }
        }
    }

    for (int r = 0; r < env->ROWS; r++) {
        for (int c = 0; c < env->COLS; c++) {
            Cell cell = env->board[BOARD_IDX(env->COLS, r, c)];
            if (cell.type != SENSOR) {
                continue;
            }
            int x = offsetX + c * CELL_SIZE;
            int y = offsetY + r * CELL_SIZE;
            int spriteIndex = cell.id % 8;
            Rectangle source = {spriteIndex * 64.0f, 529.0f, 64.0f, 30.0f};
            Rectangle dest = {x + 12.0f, y + 24.0f, 56.0f, 26.0f};
            DrawTexturePro(client->sprites, source, dest, (Vector2){0}, 0.0f, WHITE);
        }
    }

    for (int r = 0; r < env->ROWS; r++) {
        for (int c = 0; c < env->COLS; c++) {
            Cell cell = env->board[BOARD_IDX(env->COLS, r, c)];
            if (cell.type != LASER) {
                continue;
            }
            int x = offsetX + c * CELL_SIZE;
            int y = offsetY + r * CELL_SIZE;
            int spriteIndex = cell.id % 8;
            Rectangle source = {spriteIndex * 64.0f, 392.0f, 64.0f, 46.0f};
            Rectangle dest = {x + CELL_SIZE / 2.0f, y + CELL_SIZE / 2.0f, 64.0f, 46.0f};
            Vector2 origin = {32.0f, 23.0f};
            float rotation = 0.0f;
            if (r == 0) {
                rotation = 90.0f;
            } else if (r == env->ROWS - 1) {
                rotation = -90.0f;
            } else if (c == env->COLS - 1) {
                rotation = 180.0f;
                source.height = -source.height;
            }
            DrawTexturePro(client->sprites, source, dest, origin, rotation, WHITE);
        }
    }

    // draw the sinks found and mirrors used
    const float fontSize = 32.0f;
    const float spacing = 1.0f;
    const char* sinksText = TextFormat("Sinks: %i/%i", env->sinks_found, env->total_sinks);
    const char* movesText = TextFormat("Moves: %i", env->moves_made);
    const char* mirrorsText = TextFormat("Mirrors: %i/%i", env->mirrors_placed, env->optimal_mirrors);
    Vector2 movesSize = MeasureTextEx(client->font, movesText, fontSize, spacing);
    Vector2 mirrorsSize = MeasureTextEx(client->font, mirrorsText, fontSize, spacing);

    DrawTextEx(client->font, sinksText, (Vector2){16, 14}, fontSize, spacing, RAYWHITE);
    DrawTextEx(client->font, movesText, (Vector2){GetScreenWidth() - movesSize.x - 16, GetScreenHeight() - fontSize - 16}, fontSize, spacing, RAYWHITE);
    DrawTextEx(client->font, mirrorsText, (Vector2){GetScreenWidth() - mirrorsSize.x - 16, 14}, fontSize, spacing, RAYWHITE);

    if (env->sinks_found == env->total_sinks) {
        const char* solvedText = "Puzzle solved! Can you do it with less mirrors?";
        if (env->mirrors_placed == env->optimal_mirrors) {
            solvedText = "Optimal solve! Press R for the next puzzle.";
        }

        const float solvedFontSize = 24.0f;
        Vector2 solvedSize = MeasureTextEx(client->font, solvedText, solvedFontSize, spacing);
        DrawTextEx(client->font, solvedText, (Vector2){(GetScreenWidth() - solvedSize.x) / 2.0f, 56}, solvedFontSize, spacing, RAYWHITE);
    }

    EndDrawing();
    puf_web_vsync();
}

// --- Native trainer (pufferl) API ---
void puf_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "n", log->n);
}

void puf_init(Env* env, Dict* kwargs) {
    env->num_agents = 1;
    env->ROWS = INIT_ROWS;
    env->COLS = INIT_COLS;
    env->max_steps = NUM_ACTIONS;
    env->owns_buffers = 0;
    env->agents[0].action_mask = NULL;
    env->agents[0].policy = 0;
    env->board = (Cell*)calloc(env->ROWS * env->COLS, sizeof(Cell));
    load_laser_puzzle_levels(env, LASER_PUZZLE_LEVELS_PATH);
}

