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


const unsigned char NOOP = 0;
const unsigned char DOWN = 1;
const unsigned char UP = 2;
const unsigned char LEFT = 3;
const unsigned char RIGHT = 4;

const unsigned char EMPTY = 0;
const unsigned char AGENT = 1;
const unsigned char TARGET = 2;

#define BUCKET_MAX_HEIGHT 1.0f
#define DOZER_MAX_V 1.0f
#define DOZER_CAPACITY 20.0f
#define BUCKET_OFFSET 2.0f
#define BUCKET_WIDTH 2.5f
#define BUCKET_LENGTH 0.8f
#define BUCKET_HEIGHT 1.0f
#define VISION 5
#define OBSERVATION_SIZE (2*VISION + 1)

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
        }
    }
}

void init(Terraform* env) {
    env->orig_map = calloc(env->size*env->size, sizeof(float));
    env->map = calloc(env->size*env->size, sizeof(float));
    env->dozers = calloc(env->num_agents, sizeof(Dozer));
    perlin_noise(env->orig_map, env->size, env->size, 1.0/128.0, 8, 0, 0, 32.0);
    env->returns = calloc(env->num_agents, sizeof(float));
}

void allocate(Terraform* env) {
    env->observations = (unsigned char*)calloc(env->size*env->size, sizeof(unsigned char));
    env->actions = (int*)calloc(5*env->num_agents, sizeof(int));
    env->rewards = (float*)calloc(env->num_agents, sizeof(float));
    env->terminals = (unsigned char*)calloc(env->num_agents, sizeof(unsigned char));
    init(env);
}

void free_initialized(Terraform* env) {
    free(env->orig_map);
    free(env->map);
    free(env->dozers);
}

void free_allocated(Terraform* env) {
    free(env->observations);
    free(env->actions);
    free(env->rewards);
    free(env->terminals);
    free_initialized(env);
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
    for (int i = 0; i < env->num_agents; i++) {
        int x_offset = env->dozers[i].x - VISION;
        int y_offset = env->dozers[i].y - VISION;
        for (int x = 0; x < 2 * VISION + 1; x++) {
            for (int y = 0; y < 2 * VISION + 1; y++) {
                if(x_offset + x < 0 || x_offset + x >= env->size || y_offset + y < 0 || y_offset + y >= env->size) {
                    continue;
                }
                env->observations[i*OBSERVATION_SIZE*OBSERVATION_SIZE + x*OBSERVATION_SIZE + y] = env->map[
                    (x_offset + x)*env->size + (y_offset + y)];
            }
        }
    }
}

void c_reset(Terraform* env) {
    memcpy(env->map, env->orig_map, env->size*env->size*sizeof(float));
    memset(env->observations, 0, 121*sizeof(unsigned char));
    env->tick = 0;

    for (int i = 0; i < env->num_agents; i++) {
        env->dozers[i] = (Dozer){0};
        env->dozers[i].x = rand() % env->size;
        env->dozers[i].y = rand() % env->size;
    }
    compute_all_observations(env);
}

void c_step(Terraform* env) {
    //printf("step\n"); 
    //printf("tick: %d\n", env->tick);
    env->tick += 1;
    if (env->tick > 512) {
        add_log(env);
        c_reset(env);
    }

    memset(env->terminals, 0, env->num_agents*sizeof(unsigned char));
    memset(env->rewards, 0, env->num_agents*sizeof(float));

    int (*actions)[5] = (int(*)[5])env->actions; 
    for (int i = 0; i < env->num_agents; i++) {
        Dozer* dozer = &env->dozers[i];
        int* atn = actions[i];
        float accel = ((float)atn[0] - 2.0f) / 2.0f; // Discrete(5) -> [-1, 1]
        float steer = ((float)atn[1] - 2.0f) / 10.0f; // Discrete(5) -> [-0.2, 0.2]
        float bucket_v = atn[2] - 1.0f; // Discrete(3) -> [-1, 1]
        float bucket_tilt = atn[3] - 1.0f; // Discrete(3) -> [-1, 1]

        float cx = dozer->x + BUCKET_OFFSET*cosf(dozer->heading);
        float cy = dozer->y + BUCKET_OFFSET*sinf(dozer->heading);

        for (int x = cx - 5; x < cx + 5; x++) {
            for (int y = cy - 5; y < cy + 5; y++) {
                if (x < 0 || x >= env->size || y < 0 || y >= env->size) {
                    continue;
                }
                float map_height = env->map[y*env->size + x];
                env->map[y*env->size + x] = 0;
                env->rewards[i] += 0.01f;
                env->returns[i] += 0.01f;
                /*
                float bucket_height_min = map_height + dozer->bucket_height;
                if (bucket_tilt > 0.0f) {
                    // Load the bucket
                    if (dozer->bucket_height >= 0.0f) {
                        continue;
                    }
                    if (dozer->load > DOZER_CAPACITY) {
                        continue;
                    }
                    if (map_height <= 1.0f) {
                        continue;
                    }
                    dozer->load += 1.0f;
                    env->map[y*env->size + x] -= 1.0f;
                } else if (bucket_tilt < 0.0f) {
                    if (dozer->load < 1.0f) {
                        continue;
                    }
                    if (dozer->bucket_height <= 0.0f) {
                        continue;
                    }
                    dozer->load -= 1.0f;
                    env->map[y*env->size + x] += 1.0f;
                }
                */
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

        dozer->v += accel;
        if (dozer->v > DOZER_MAX_V) {
            dozer->v = DOZER_MAX_V;
        }
        if (dozer->v < -DOZER_MAX_V) {
            dozer->v = -DOZER_MAX_V;
        }

        dozer->bucket_height += bucket_v;
        if (dozer->bucket_height > BUCKET_MAX_HEIGHT) {
            dozer->bucket_height = BUCKET_MAX_HEIGHT;
        }
        if (dozer->bucket_height < -BUCKET_MAX_HEIGHT) {
            dozer->bucket_height = -BUCKET_MAX_HEIGHT;
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
    UploadMesh(mesh, false);
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
    Texture2D texture;
    Model dozer;
};

Client* make_client(Terraform* env) {
    Client* client = (Client*)calloc(1, sizeof(Client));
    int px = 64*env->size;
    InitWindow(1080, 720, "PufferLib Terraform");
    SetTargetFPS(30);
    Camera3D camera = { 0 };
    camera.position = (Vector3){ 450.0f, 275.0f, 530.0f }; // Camera position
    camera.target = (Vector3){ 0.0f, 0.0f, 0.0f };      // Camera looking at point
    camera.up = (Vector3){ 0.0f, 1.0f, 0.0f };          // Camera up vector (rotation towards target)
    camera.fovy = 45.0f;                                // Camera field-of-view Y
    camera.projection = CAMERA_PERSPECTIVE;             // Camera projection type
    client->camera = camera;

    //Image checked = GenImageChecked(env->size, env->size, 2, 2, PUFF_RED, PUFF_CYAN);
    Image img = LoadImage("resources/terraform/perlin.jpg");
    client->texture = LoadTextureFromImage(img);
    client->dozer = LoadModel("resources/terraform/dozer.glb");
    UnloadImage(img);
    return client;
}

void close_client(Client* client) {
    CloseWindow();
    free(client->mesh);
    free(client);
}

void c_render(Terraform* env) {
    if (env->client == NULL) {
        env->client = make_client(env);
    }
    if (IsKeyDown(KEY_ESCAPE)) {
        exit(0);
    }
    Client* client = env->client;

    if (client->mesh == NULL) {
        UnloadModel(client->model);
        //UnloadMesh(*client->mesh);
    }
    client->mesh = create_heightmap_mesh(env->map, (Vector3){env->size, 1, env->size});
    update_heightmap_mesh(client->mesh, env->map, (Vector3){env->size, 1, env->size});
    client->model = LoadModelFromMesh(*client->mesh);
    client->model.materials[0].maps[MATERIAL_MAP_DIFFUSE].texture = client->texture;

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
    DrawModel(client->model, (Vector3){0, 0, 0}, 1.0f, (Color){156, 50, 20, 255});
    for (int i = 0; i < env->num_agents; i++) {
        Dozer* dozer = &env->dozers[i];
        int x = (int)dozer->x;
        int z = (int)dozer->y;  
        int size = (int)env->size;
        
        // Get height from map using correct indexing
        float y = env->map[z * size + x] + 0.5f;
        rlPushMatrix();
        rlTranslatef(dozer->x, y, dozer->y);
        rlRotatef(-90.f - dozer->heading*RAD2DEG, 0, 1, 0);
        DrawModel(client->dozer, (Vector3){0, 0, 0}, 1.0f, WHITE);
        rlPopMatrix();
        // DrawCube((Vector3){dozer->x, y, dozer->y}, 1.0f, 1.0f, 1.0f, PUFF_WHITE);
    }
    EndMode3D();
    DrawText(TextFormat("Camera x: %f", client->camera.position.x), 10, 150, 20, PUFF_WHITE);
    DrawText(TextFormat("Camera y: %f", client->camera.position.y), 10, 170, 20, PUFF_WHITE);
    DrawText(TextFormat("Camera z: %f", client->camera.position.z), 10, 190, 20, PUFF_WHITE);
    EndDrawing();
}
