// new_racing.h
#include <stdlib.h>
#include <math.h>

#define NOOP 0
#define ACCEL 1
#define DECEL 2
#define LEFT 3
#define RIGHT 4
#define MAX_SPEED 100.0
#define MIN_SPEED -10.0
#define SPEED_INCREMENT 5.0
#define ROAD_WIDTH 90.0
#define CAR_WIDTH 16.0
#define PLAYER_CAR_LENGTH 11.0
#define ENEMY_CAR_LENGTH 11.0
#define MAX_Y_POSITION 155
#define MIN_Y_POSITION 0
#define PASS_THRESHOLD 155

typedef struct CRacingEnv CRacingEnv;
struct CRacingEnv {
    float* observations;
    unsigned char* actions;
    float* rewards;
    unsigned char* dones;
    int dayNumber;
    int cars_passed;
    int tick;
    int day_length;
    float speed;
    int* carsRemainingLo;
    int* carsRemainingHi;
    float* enemy_car_y;
    float* enemy_car_x;
    int* enemy_active;
    int num_enemy_cars;
    int done;
};

void init_racing_env(CRacingEnv* env) {
    env->speed = MIN_SPEED;
    env->dayNumber = 1;
    env->cars_passed = 0;
    env->day_length = 300000;
    env->tick = 0;
    env->done = 0;

    for (int i = 0; i < 15; i++) {
        env->enemy_car_y[i] = MAX_Y_POSITION + 100.0;
        env->enemy_active[i] = 0;
    }
}

void step(CRacingEnv* env) {
    env->speed += env->actions[0] == ACCEL ? SPEED_INCREMENT : 0;
    env->speed -= env->actions[0] == DECEL ? SPEED_INCREMENT : 0;
    if (env->speed > MAX_SPEED) env->speed = MAX_SPEED;
    if (env->speed < MIN_SPEED) env->speed = MIN_SPEED;

    for (int i = 0; i < env->num_enemy_cars; i++) {
        if (env->enemy_active[i]) {
            env->enemy_car_y[i] -= env->speed * 0.1;

            if (env->enemy_car_y[i] < MIN_Y_POSITION - ENEMY_CAR_LENGTH) {
                env->enemy_active[i] = 0;
                env->cars_passed++;
            }
        }
    }

    env->tick++;
    if (env->tick >= env->day_length) {
        env->done = 1;
    }
}

void reset(CRacingEnv* env) {
    env->tick = 0;
    env->speed = MIN_SPEED;
    env->cars_passed = 0;

    for (int i = 0; i < env->num_enemy_cars; i++) {
        env->enemy_active[i] = 0;
    }
}

void spawn_enemy_cars(CRacingEnv* env) {
    for (int i = 0; i < env->num_enemy_cars; i++) {
        if (env->enemy_active[i] == 0) {
            env->enemy_car_y[i] = MAX_Y_POSITION + (rand() % 500);
            env->enemy_active[i] = 1;
        }
    }
}

void free_racing_env(CRacingEnv* env) {
    free(env->observations);
    free(env->enemy_car_y);
    free(env->enemy_car_x);
    free(env->enemy_active);
}
