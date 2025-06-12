#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <limits.h>
#include <float.h>
#include <time.h>
#include "raylib.h"
#include "simplex.h"
#include "raymath.h"
#include "rlgl.h"

#if defined(PLATFORM_DESKTOP)
    #define GLSL_VERSION 330
#else
    #define GLSL_VERSION 100
#endif

const unsigned char NOOP = 0;
const unsigned char DOWN = 1;
const unsigned char UP = 2;
const unsigned char LEFT = 3;
const unsigned char RIGHT = 4;

const unsigned char EMPTY = 0;
const unsigned char AGENT = 1;
const unsigned char TARGET = 2;

#define MAX_DIRT_HEIGHT 32.0f
#define BUCKET_MAX_HEIGHT 1.0f
#define DOZER_MAX_V 2.0f
#define DOZER_CAPACITY 200.0f
#define BUCKET_OFFSET 5.0f
#define BUCKET_WIDTH 2.5f
#define BUCKET_LENGTH 0.8f
#define BUCKET_HEIGHT 1.0f
#define VISION 5
#define OBSERVATION_SIZE (2*VISION + 1)
#define TOTAL_OBS (OBSERVATION_SIZE*OBSERVATION_SIZE + 4)
#define DOZER_STEP_HEIGHT 5.0f

typedef struct Log Log;
struct Log {
    float perf;
    float score;
    float episode_return;
    float episode_length;
    float n;
};

typedef struct Dozer {
    float x;
    float y;
    float z;
    float v;
    float heading;
    float bucket_height;
    float bucket_tilt;
    float load;
} Dozer;
 
typedef struct Client Client;
typedef struct Terraform {
    Log log;
    Client* client;
    Dozer* dozers;
    unsigned char* observations;
    int* actions;
    float* rewards;
    float* returns;
    unsigned char* terminals;
    int size;
    int tick;
    float* orig_map;
    float* map;
    float* target_map;
    int num_agents;
} Terraform;

float randf(float min, float max) {
    return min + (max - min)*(float)rand()/(float)RAND_MAX;
}

void perlin_noise(float* map, int width, int height,
        float base_frequency, int octaves, int offset_x, int offset_y, float glob_scale) {
    float frequencies[octaves];
    for (int i = 0; i < octaves; i++) {
        frequencies[i] = base_frequency*pow(2, i);
    }

    float min_value = FLT_MAX;
    float max_value = FLT_MIN;
    for (int r = 0; r < height; r++) {
        for (int c = 0; c < width; c++) {
            int adr = r*width + c;
            for (int oct = 0; oct < octaves; oct++) {
                float freq = frequencies[oct];
                map[adr] += (1.0/pow(2, oct))*noise2(freq*c + offset_x, freq*r + offset_y);
            }
            float val = map[adr];
            if (val < min_value) {
                min_value = val;
            }
            if (val > max_value) {
                max_value = val;
            }
        }
    }

    float scale = 1.0/(max_value - min_value);
    for (int r = 0; r < height; r++) {
        for (int c = 0; c < width; c++) {
            int adr = r*width + c;
            map[adr] = glob_scale * scale * (map[adr] - min_value);
            if (map[adr] < 64.0f) {
                map[adr] = 0.0f;
            } else {
                map[adr] -= 64.0f;
            }
        }
    }
}

int map_idx(Terraform* env, float x, float y) {
    return env->size*(int)y + (int)x;
}

void init(Terraform* env) {
    env->orig_map = calloc(env->size*env->size, sizeof(float));
    env->map = calloc(env->size*env->size, sizeof(float));
    env->target_map = calloc(env->size*env->size, sizeof(float));
    //for (int i = 0; i < env->size*env->size; i++) {
    //    env->target_map[i] = 1.0f;
    // }
    env->dozers = calloc(env->num_agents, sizeof(Dozer));
    perlin_noise(env->orig_map, env->size, env->size, 1.0/128.0, 8, 0, 0, MAX_DIRT_HEIGHT+64);
    perlin_noise(env->target_map, env->size, env->size, 1.0/128.0, 8, env->size, env->size, MAX_DIRT_HEIGHT+64);
    env->returns = calloc(env->num_agents, sizeof(float));
}

void free_initialized(Terraform* env) {
    free(env->orig_map);
    free(env->map);
    free(env->dozers);
}

void add_log(Terraform* env) {
    for (int i = 0; i < env->num_agents; i++) {
        env->log.perf += env->returns[i];
        env->log.score += env->returns[i];
        env->log.episode_length += env->tick;
        env->log.episode_return += env->returns[i];
        env->log.n++;
    }
}

void compute_all_observations(Terraform* env) {
    int dialate = 4;
    int obs_idx = 0;
    for (int i = 0; i < env->num_agents; i++) {
        int x_offset = env->dozers[i].x - dialate*VISION;
        int y_offset = env->dozers[i].y - dialate*VISION;
        for (int x = 0; x < 2*dialate*VISION + 1; x+=dialate) {
            for (int y = 0; y < 2*dialate*VISION + 1; y+=dialate) {
                if(x_offset + x < 0 || x_offset + x >= env->size || y_offset + y < 0 || y_offset + y >= env->size) {
                    env->observations[obs_idx++] = 0;
                    env->observations[obs_idx++] = 0;
                    continue;
                }
                int idx = (x_offset + x)*env->size + (y_offset + y);
                env->observations[obs_idx++] = env->map[idx] * (255.0f/MAX_DIRT_HEIGHT);
                env->observations[obs_idx++] = 127.0 + (env->target_map[idx] - env->map[idx]) * (128.0f/MAX_DIRT_HEIGHT);
            }
        }
        Dozer* dozer = &env->dozers[i];
        env->observations[obs_idx++] = (255.0f*dozer->x)/env->size;
        env->observations[obs_idx++] = (255.0f*dozer->y)/env->size;
        env->observations[obs_idx++] = 20.0f*(dozer->v + DOZER_MAX_V);
        // This is -5?
        env->observations[obs_idx++] = 20.0f*(dozer->heading + 2*PI);
    }
}

void c_reset(Terraform* env) {
    memcpy(env->map, env->orig_map, env->size*env->size*sizeof(float));
    memset(env->observations, 0, env->num_agents*246*sizeof(unsigned char));
    memset(env->returns, 0, env->num_agents*sizeof(float));
    env->tick = 0;

    for (int i = 0; i < env->num_agents; i++) {
        env->dozers[i] = (Dozer){0};
        do {
            env->dozers[i].x = rand() % env->size;
            env->dozers[i].y = rand() % env->size;
        } while (env->map[map_idx(env, env->dozers[i].x, env->dozers[i].y)] != 0.0f);
    }
    compute_all_observations(env);
}

void c_step(Terraform* env) {
    //printf("step\n"); 
    //printf("tick: %d\n", env->tick);
    env->tick += 1;
    if (rand() % 5550 == 0) {
        add_log(env);
        c_reset(env);
    }

    memset(env->terminals, 0, env->num_agents*sizeof(unsigned char));
    memset(env->rewards, 0, env->num_agents*sizeof(float));

    int (*actions)[3] = (int(*)[3])env->actions; 
    for (int i = 0; i < env->num_agents; i++) {
        Dozer* dozer = &env->dozers[i];
        int* atn = actions[i];
        float accel = ((float)atn[0] - 2.0f) / 2.0f; // Discrete(5) -> [-1, 1]
        float steer = ((float)atn[1] - 2.0f) / 10.0f; // Discrete(5) -> [-0.2, 0.2]
        int bucket_atn = atn[2];

        float cx = dozer->x + BUCKET_OFFSET*cosf(dozer->heading);
        float cy = dozer->y + BUCKET_OFFSET*sinf(dozer->heading);

        for (int x = cx - 2; x < cx + 2; x++) {
            for (int y = cy - 2; y < cy + 2; y++) {
                if (x < 0 || x >= env->size || y < 0 || y >= env->size) {
                    continue;
                }
                int idx = map_idx(env, x, y);
                float map_height = env->map[idx];
                float target_height = env->target_map[idx];
                float delta_pre = fabsf(map_height - target_height);

                if (bucket_atn == 0) {
                    continue;
                } else if (bucket_atn == 1) { // Load
                    // Can't load while backing up
                    if (dozer->v < 0) {
                        continue;
                    }

                    // Load up to 1 unit of dirt
                    float load_amount = 1.0f;
                    if (map_height <= 1.0f) {
                        load_amount = map_height;
                    }

                    // Don't overload the bucket
                    if (dozer->load + load_amount > DOZER_CAPACITY) {
                        load_amount = DOZER_CAPACITY - dozer->load;
                    }

                    dozer->load += load_amount;
                    env->map[idx] -= load_amount;
                    map_height -= load_amount;
                } else if (bucket_atn == 2) { // Unload
                    // Can't unload while moving forward
                    if (dozer->v > 0) {
                        continue;
                    }

                    float unload_amount = 1.0f;
                    if (dozer->load < unload_amount) {
                        unload_amount = dozer->load;
                    }

                    if (map_height + unload_amount > MAX_DIRT_HEIGHT) {
                        unload_amount = MAX_DIRT_HEIGHT - map_height;
                    }

                    dozer->load -= unload_amount;
                    env->map[idx] += unload_amount;
                    map_height += unload_amount;
                }

                // Reward for terraforming towards target map
                float delta_post = fabsf(map_height - target_height);
                float reward = 0.01*(delta_pre - delta_post);
                env->rewards[i] += reward;
                env->returns[i] += reward;
            }
        }

        // Bucket AABB
        /*
        float x_min = bucket_cx - BUCKET_WIDTH/2.0f*cosf(dozer->heading) + BUCKET_LENGTH/2.0f*sinf(dozer->heading);
        float x_max = bucket_cx + BUCKET_WIDTH/2.0f*cosf(dozer->heading) + BUCKET_LENGTH/2.0f*sinf(dozer->heading);
        float y_min = bucket_cy - BUCKET_WIDTH/2.0f*sinf(dozer->heading) + BUCKET_LENGTH/2.0f*cosf(dozer->heading);
        float y_max = bucket_cy + BUCKET_WIDTH/2.0f*sinf(dozer->heading) + BUCKET_LENGTH/2.0f*cosf(dozer->heading);

        for (int x = x_min; x < x_max; x++) {
            for (int y = y_min; y < y_max; y++) {
                float cell_x = x + 0.5f;
                float cell_y = y + 0.5f;

            }
        }
        */

        dozer->heading += steer;
        if (dozer->heading > 2*PI) {
            dozer->heading -= 2*PI;
        }
        if (dozer->heading < 0) {
            dozer->heading += 2*PI;
        }

        dozer->v += accel;
        if (dozer->v > DOZER_MAX_V) {
            dozer->v = DOZER_MAX_V;
        }
        if (dozer->v < -DOZER_MAX_V) {
            dozer->v = -DOZER_MAX_V;
        }

        int idx = map_idx(env, dozer->x, dozer->y);
        float dozer_height = env->map[idx];

        // Raytrace collision
        for (int d=0; d<dozer->v; d++) {
            float x = dozer->x + d*cosf(dozer->heading);
            float y = dozer->y + d*sinf(dozer->heading);
            if (x < 0 || x >= env->size-1 || y < 0 || y >= env->size-1) {
                continue;
            }

            int dst_idx = map_idx(env, x, y);
            float dst_height = env->map[dst_idx];
            if (fabsf(dozer_height - dst_height) > DOZER_STEP_HEIGHT) {
                dozer->v = 0;
            }
        }

        // Box collision around final destination
        float dst_x = dozer->x + dozer->v*cosf(dozer->heading);
        float dst_y = dozer->y + dozer->v*sinf(dozer->heading);
        for (int x=(int)(dst_x-1.0f); x<=(int)(dst_x+1.0f); x++) {
            for (int y=(int)(dst_y-1.0f); y<=(int)(dst_y+1.0f); y++) {
                if (x < 0 || x >= env->size-1 || y < 0 || y >= env->size-1) {
                    continue;
                }
                int dst_idx = map_idx(env, x, y);
                float dst_height = env->map[dst_idx];
                if (fabsf(dozer_height - dst_height) > DOZER_STEP_HEIGHT) {
                    dozer->v = 0;
                }
            }
        }

        dozer->x += dozer->v*cosf(dozer->heading);
        dozer->y += dozer->v*sinf(dozer->heading);

        if (dozer->x < 0) {
            dozer->x = 0;
        }
        if (dozer->x >= env->size) {
            dozer->x = env->size - 1;
        }
        if (dozer->y < 0) {
            dozer->y = 0;
        }
        if (dozer->y >= env->size) {
            dozer->y = env->size - 1;
        }

        // Teleportitis
        if (rand() % 512 == 0) {
            do {
                env->dozers[i].x = rand() % env->size;
                env->dozers[i].y = rand() % env->size;
            } while (env->map[map_idx(env, env->dozers[i].x, env->dozers[i].y)] != 0.0f);
        }
 
    }

    //printf("observations\n");
    compute_all_observations(env);
    //int action = env->actions[0];
}

void c_close(Terraform* env) {
}

Mesh* create_heightmap_mesh(float* heightMap, Vector3 size) {
    int mapX = size.x;
    int mapZ = size.z;

    // NOTE: One vertex per pixel
    Mesh* mesh = (Mesh*)calloc(1, sizeof(Mesh));
    mesh->triangleCount = (mapX - 1)*(mapZ - 1)*2;    // One quad every four pixels

    mesh->vertexCount = mesh->triangleCount*3;

    mesh->vertices = (float *)RL_MALLOC(mesh->vertexCount*3*sizeof(float));
    mesh->normals = (float *)RL_MALLOC(mesh->vertexCount*3*sizeof(float));
    mesh->texcoords = (float *)RL_MALLOC(mesh->vertexCount*2*sizeof(float));
    mesh->colors = NULL;
    UploadMesh(mesh, false);
    return mesh;
}

void update_heightmap_mesh(Mesh* mesh, float* heightMap, Vector3 size) {
    int mapX = size.x;
    int mapZ = size.z;

    int vCounter = 0;       // Used to count vertices float by float
    int tcCounter = 0;      // Used to count texcoords float by float
    int nCounter = 0;       // Used to count normals float by float

    //Vector3 scaleFactor = { size.x/(mapX - 1), 1.0f, size.z/(mapZ - 1) };
    Vector3 scaleFactor = { 1.0f, 1.0f, 1.0f};

    Vector3 vA = { 0 };
    Vector3 vB = { 0 };
    Vector3 vC = { 0 };
    Vector3 vN = { 0 };

    for (int z = 0; z < mapZ-1; z++)
    {
        for (int x = 0; x < mapX-1; x++)
        {
            // Fill vertices array with data
            //----------------------------------------------------------

            // one triangle - 3 vertex
            mesh->vertices[vCounter] = (float)x*scaleFactor.x;
            mesh->vertices[vCounter + 1] = heightMap[x + z*mapX]*scaleFactor.y;
            mesh->vertices[vCounter + 2] = (float)z*scaleFactor.z;

            mesh->vertices[vCounter + 3] = (float)x*scaleFactor.x;
            mesh->vertices[vCounter + 4] = heightMap[x + (z + 1)*mapX]*scaleFactor.y;
            mesh->vertices[vCounter + 5] = (float)(z + 1)*scaleFactor.z;

            mesh->vertices[vCounter + 6] = (float)(x + 1)*scaleFactor.x;
            mesh->vertices[vCounter + 7] = heightMap[(x + 1) + z*mapX]*scaleFactor.y;
            mesh->vertices[vCounter + 8] = (float)z*scaleFactor.z;

            // Another triangle - 3 vertex
            mesh->vertices[vCounter + 9] = mesh->vertices[vCounter + 6];
            mesh->vertices[vCounter + 10] = mesh->vertices[vCounter + 7];
            mesh->vertices[vCounter + 11] = mesh->vertices[vCounter + 8];

            mesh->vertices[vCounter + 12] = mesh->vertices[vCounter + 3];
            mesh->vertices[vCounter + 13] = mesh->vertices[vCounter + 4];
            mesh->vertices[vCounter + 14] = mesh->vertices[vCounter + 5];

            mesh->vertices[vCounter + 15] = (float)(x + 1)*scaleFactor.x;
            mesh->vertices[vCounter + 16] = heightMap[(x + 1) + (z + 1)*mapX]*scaleFactor.y;
            mesh->vertices[vCounter + 17] = (float)(z + 1)*scaleFactor.z;
            vCounter += 18;     // 6 vertex, 18 floats

            // Fill texcoords array with data
            //--------------------------------------------------------------
            mesh->texcoords[tcCounter] = (float)x/(mapX - 1);
            mesh->texcoords[tcCounter + 1] = (float)z/(mapZ - 1);

            mesh->texcoords[tcCounter + 2] = (float)x/(mapX - 1);
            mesh->texcoords[tcCounter + 3] = (float)(z + 1)/(mapZ - 1);

            mesh->texcoords[tcCounter + 4] = (float)(x + 1)/(mapX - 1);
            mesh->texcoords[tcCounter + 5] = (float)z/(mapZ - 1);

            mesh->texcoords[tcCounter + 6] = mesh->texcoords[tcCounter + 4];
            mesh->texcoords[tcCounter + 7] = mesh->texcoords[tcCounter + 5];

            mesh->texcoords[tcCounter + 8] = mesh->texcoords[tcCounter + 2];
            mesh->texcoords[tcCounter + 9] = mesh->texcoords[tcCounter + 3];

            mesh->texcoords[tcCounter + 10] = (float)(x + 1)/(mapX - 1);
            mesh->texcoords[tcCounter + 11] = (float)(z + 1)/(mapZ - 1);
            tcCounter += 12;    // 6 texcoords, 12 floats

            // Fill normals array with data
            //--------------------------------------------------------------
            for (int i = 0; i < 18; i += 9)
            {
                vA.x = mesh->vertices[nCounter + i];
                vA.y = mesh->vertices[nCounter + i + 1];
                vA.z = mesh->vertices[nCounter + i + 2];

                vB.x = mesh->vertices[nCounter + i + 3];
                vB.y = mesh->vertices[nCounter + i + 4];
                vB.z = mesh->vertices[nCounter + i + 5];

                vC.x = mesh->vertices[nCounter + i + 6];
                vC.y = mesh->vertices[nCounter + i + 7];
                vC.z = mesh->vertices[nCounter + i + 8];

                vN = Vector3Normalize(Vector3CrossProduct(Vector3Subtract(vB, vA), Vector3Subtract(vC, vA)));

                mesh->normals[nCounter + i] = vN.x;
                mesh->normals[nCounter + i + 1] = vN.y;
                mesh->normals[nCounter + i + 2] = vN.z;

                mesh->normals[nCounter + i + 3] = vN.x;
                mesh->normals[nCounter + i + 4] = vN.y;
                mesh->normals[nCounter + i + 5] = vN.z;

                mesh->normals[nCounter + i + 6] = vN.x;
                mesh->normals[nCounter + i + 7] = vN.y;
                mesh->normals[nCounter + i + 8] = vN.z;
            }

            nCounter += 18;     // 6 vertex, 18 floats
        }
    }

    // Upload vertex data to GPU (static mesh)
    UpdateMeshBuffer(*mesh, 0, mesh->vertices, mesh->vertexCount * 3 * sizeof(float), 0); // Update vertices
    UpdateMeshBuffer(*mesh, 2, mesh->normals, mesh->vertexCount * 3 * sizeof(float), 0); // Update normals
}

const Color PUFF_RED = (Color){187, 0, 0, 255};
const Color PUFF_CYAN = (Color){0, 187, 187, 255};
const Color PUFF_WHITE = (Color){241, 241, 241, 241};
const Color PUFF_BACKGROUND = (Color){6, 24, 24, 255};
const Color PUFF_BACKGROUND2 = (Color){18, 72, 72, 255};

typedef struct Client Client;
struct Client {
    Texture2D ball;
    Camera3D camera;
    Mesh* mesh;
    Model model;
    Mesh* target_mesh;
    Model target_model;
    Texture2D texture;
    Model dozer;
    Shader shader;
    Shader target_shader;
    Texture2D shader_terrain;
    int shader_terrain_loc;
    unsigned char *shader_terrain_data;
};

Client* make_client(Terraform* env) {
    Client* client = (Client*)calloc(1, sizeof(Client));
    InitWindow(1080, 720, "PufferLib Terraform");
    SetConfigFlags(FLAG_MSAA_4X_HINT);
    SetTargetFPS(600);
    Camera3D camera = { 0 };
                                                       //
    camera.up = (Vector3){ 0.0f, 1.0f, 0.0f };          // Camera up vector (rotation towards target)
    camera.fovy = 45.0f;                                // Camera field-of-view Y
    camera.projection = CAMERA_PERSPECTIVE;             // Camera projection type
    camera.position = (Vector3){ 3*env->size/4, env->size, 3*env->size/4};
    camera.target = (Vector3){ env->size/2, 0, env->size/2-1};
    client->camera = camera;

    client->shader = LoadShader(
        TextFormat("resources/terraform/shader_%i.vs", GLSL_VERSION),
        TextFormat("resources/terraform/shader_%i.fs", GLSL_VERSION)
    );
    client->target_shader = LoadShader(
        TextFormat("resources/terraform/target_shader_%i.vs", GLSL_VERSION),
        TextFormat("resources/terraform/target_shader_%i.fs", GLSL_VERSION)
    );

    Image img = GenImageColor(env->size, env->size, WHITE);
    ImageFormat(&img, PIXELFORMAT_UNCOMPRESSED_R8G8B8A8);
    client->shader_terrain = LoadTextureFromImage(img);
    UnloadImage(img);

    client->shader_terrain_loc = GetShaderLocation(client->target_shader, "terrain");
    SetShaderValueTexture(client->target_shader, client->shader_terrain_loc, client->shader_terrain);

    client->shader_terrain_data = calloc(4*env->size*env->size, sizeof(unsigned char));

    int shader_width_loc = GetShaderLocation(client->target_shader, "width");
    SetShaderValue(client->target_shader, shader_width_loc, &env->size, SHADER_UNIFORM_INT);

    int shader_height_loc = GetShaderLocation(client->target_shader, "height");
    SetShaderValue(client->target_shader, shader_height_loc, &env->size, SHADER_UNIFORM_INT);
 
    //Image checked = GenImageChecked(env->size, env->size, 2, 2, PUFF_RED, PUFF_CYAN);
    img = LoadImage("resources/terraform/perlin.jpg");
    client->texture = LoadTextureFromImage(img);
    client->dozer = LoadModel("resources/terraform/dozer.glb");
    UnloadImage(img);
    client->mesh = NULL;
    return client;
}

void close_client(Client* client) {
    CloseWindow();
    free(client->mesh);
    free(client);
}

void handle_camera_controls(Client* client) {
    static Vector2 prev_mouse_pos = {0};
    static bool is_dragging = false;
    float camera_move_speed = 0.5f;

    // Handle mouse drag for camera movement
    if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) {
        prev_mouse_pos = GetMousePosition();
        is_dragging = true;
    }

    if (IsMouseButtonReleased(MOUSE_BUTTON_LEFT)) {
        is_dragging = false;
    }

    if (is_dragging) {
        Vector2 current_mouse_pos = GetMousePosition();
        Vector2 delta = {
            -(current_mouse_pos.x - prev_mouse_pos.x) * camera_move_speed,
            (current_mouse_pos.y - prev_mouse_pos.y) * camera_move_speed
        };

        // Apply 45-degree rotation to the movement
        // For a -45 degree rotation (clockwise)
        float cos45 = -0.7071f;  // cos(-45°)
        float sin45 = 0.7071f; // sin(-45°)
        Vector2 rotated_delta = {
            delta.x * cos45 - delta.y * sin45,
            delta.x * sin45 + delta.y * cos45
        };

        // Update camera position (only X and Y)
        client->camera.position.z += rotated_delta.x;
        client->camera.position.x += rotated_delta.y;

        // Update camera target (only X and Y)
        client->camera.target.z += rotated_delta.x;
        client->camera.target.x += rotated_delta.y;

        prev_mouse_pos = current_mouse_pos;
    }

    // Handle mouse wheel for zoom
    float wheel = GetMouseWheelMove();
    if (wheel != 0) {
        float zoom_factor = 1.0f - (wheel * 0.1f);
        // Calculate the current direction vector from target to position
        Vector3 direction = {
            client->camera.position.x - client->camera.target.x,
            client->camera.position.y - client->camera.target.y,
            client->camera.position.z - client->camera.target.z
        };

        // Scale the direction vector by the zoom factor
        direction.x *= zoom_factor;
        direction.y *= zoom_factor;
        direction.z *= zoom_factor;

        // Update the camera position based on the scaled direction
        client->camera.position.x = client->camera.target.x + direction.x;
        client->camera.position.y = client->camera.target.y + direction.y;
        client->camera.position.z = client->camera.target.z + direction.z;
    }
}

void c_render(Terraform* env) {
    if (env->client == NULL) {
        env->client = make_client(env);
        env->client->mesh = create_heightmap_mesh(env->map, (Vector3){env->size, 1, env->size});
        update_heightmap_mesh(env->client->mesh, env->map, (Vector3){env->size, 1, env->size});
        env->client->model = LoadModelFromMesh(*env->client->mesh);

        env->client->target_mesh = create_heightmap_mesh(env->target_map, (Vector3){env->size, 1, env->size});
        update_heightmap_mesh(env->client->target_mesh, env->target_map, (Vector3){env->size, 1, env->size});
        env->client->target_model = LoadModelFromMesh(*env->client->target_mesh);
    }
    if (IsKeyDown(KEY_ESCAPE)) {
        exit(0);
    }
    Client* client = env->client;

    handle_camera_controls(client);
    //Camera3D* camera = &client->camera;
    //camera->position = (Vector3){ x+30, z+100.0f, y+30 };
    //camera->target = (Vector3){ x, 0, y-1};
 
    if (env->tick % 10 == 0) {
        update_heightmap_mesh(client->mesh, env->map, (Vector3){env->size, 1, env->size});
        update_heightmap_mesh(client->target_mesh, env->target_map, (Vector3){env->size, 1, env->size});
        for (int i = 0; i < env->size*env->size; i++) {
            client->shader_terrain_data[4*i] = env->map[i];
            client->shader_terrain_data[4*i+3] = 255;
        }
        UpdateTexture(client->shader_terrain, client->shader_terrain_data);
        SetShaderValueTexture(client->target_shader, env->client->shader_terrain_loc, env->client->shader_terrain);

    }
    //client->model.materials[0].maps[MATERIAL_MAP_DIFFUSE].texture = client->texture;
    client->model.materials[0].shader = client->shader;

    //client->target_model.materials[0].maps[MATERIAL_MAP_DIFFUSE].texture = client->texture;
    client->target_model.materials[0].shader = client->target_shader;

    //update_heightmap_mesh(client->mesh, env->map, (Vector3){env->size, 1, env->size});
    //client->model = LoadModelFromMesh(*client->mesh);

    BeginDrawing();
    ClearBackground((Color){143, 86, 29, 255});
    BeginMode3D(client->camera);
    /*
    for(int i = 0; i < env->size*env->size; i++) {
        float height = env->map[i];
        int x = i%env->size;
        int z = i/env->size;
        DrawCube((Vector3){x, height, z}, 1.0f, 1.0f, 1.0f, DARKGREEN);
        DrawCubeWires((Vector3){x, height, z}, 1.0f, 1.0f, 1.0f, MAROON);
    }
    */

    BeginShaderMode(client->shader);
    DrawModel(client->model, (Vector3){0, 0, 0}, 1.0f, (Color){156, 50, 20, 255});
    EndShaderMode();

    BeginShaderMode(client->target_shader);
    DrawModel(client->target_model, (Vector3){0, 0, 0}, 1.0f, (Color){156, 50, 20, 255});
    EndShaderMode();

    for (int i = 0; i < env->num_agents; i++) {
        Dozer* dozer = &env->dozers[i];
        int x = (int)dozer->x;
        int z = (int)dozer->y;  
        int size = (int)env->size;
        
        // Get height from map using correct indexing
        float y = env->map[z * size + x] + 0.5f;
        float yy = y;
        rlPushMatrix();
        rlTranslatef(dozer->x, y, dozer->y);
        rlRotatef(-90.f - dozer->heading*RAD2DEG, 0, 1, 0);
        DrawModel(client->dozer, (Vector3){0, 0, 0}, 0.25f, WHITE);
        rlPopMatrix();
        // DrawCube((Vector3){dozer->x, y, dozer->y}, 1.0f, 1.0f, 1.0f, PUFF_WHITE);
        int dialate = 4;
        int x_offset = env->dozers[i].x - dialate*VISION;
        int y_offset = env->dozers[i].y - dialate*VISION;
        for (int x = 0; x < 2*dialate*VISION + 1; x+=dialate) {
            for (int y = 0; y < 2*dialate*VISION + 1; y+=dialate) {
                if(x_offset + x < 0 || x_offset + x >= env->size || y_offset + y < 0 || y_offset + y >= env->size) {
                    continue;
                }
                //DrawCube((Vector3){x_offset + x, yy, y_offset + y}, 0.5f, 0.5f, 0.5f, PUFF_WHITE);
            }
        }
    }
    EndMode3D();
    //DrawText(TextFormat("Dozer x: %f", x), 10, 150, 20, PUFF_WHITE);
    //DrawText(TextFormat("Dozer y: %f", y), 10, 170, 20, PUFF_WHITE);
    //DrawText(TextFormat("Dozer z: %f", z), 10, 190, 20, PUFF_WHITE);
    DrawFPS(10, 10);
    EndDrawing();
}
