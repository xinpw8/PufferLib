#include "raylib.h"
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define BOARD_SIZE 11
#define TOTAL_CELLS (BOARD_SIZE * BOARD_SIZE)
#define PLAYER_COLOR 1
#define ENV_COLOR -1

// Virtual nodes for incremental connection tracking
#define TOP_NODE TOTAL_CELLS
#define BOTTOM_NODE (TOTAL_CELLS + 1)
#define LEFT_NODE (TOTAL_CELLS + 2)
#define RIGHT_NODE (TOTAL_CELLS + 3)
#define TOTAL_NODES (TOTAL_CELLS + 4)

const int dr[] = { -1, -1, 0, 0, 1, 1 };
const int dc[] = { 0, 1, -1, 1, -1, 0 };

// Required struct. Only use floats!
typedef struct {
	float perf; // Recommended 0-1 normalized single real number perf metric
	float score; // Recommended unnormalized single real number perf metric
	float episode_return; // Recommended metric: sum of agent rewards over episode
	float episode_length; // Recommended metric: number of steps of agent episode
	// Any extra fields you add here may be exported in binding.c
	float n; // Required as the last field
} Log;

// Required that you have some struct for your env
typedef struct {
	Log log; // Required field. Env binding code uses this to aggregate logs
	float* observations; // Required. You can use any obs type, but make sure it matches in Python!
	float* actions; // Required
	float* rewards; // Required
	float* terminals; // Required
	int num_agents;
	int tick;
	int current_player;
	int8_t board[TOTAL_CELLS];
	bool random_opponent;

	// Disjoint Set Union (Union-Find) tracking arrays
	int parent[TOTAL_NODES];
	int size[TOTAL_NODES];

	unsigned int rng;
} Hex;

void allocate_chex(Hex* env)
{
	env->observations = (float*)calloc(2 * TOTAL_CELLS, sizeof(float));
	env->actions = (float*)calloc(1, sizeof(float));
	env->terminals = (float*)calloc(1, sizeof(float));
	env->rewards = (float*)calloc(1, sizeof(float));
}

void free_allocated_chex(Hex* env)
{
	free(env->actions);
	free(env->observations);
	free(env->terminals);
	free(env->rewards);
}

void init(Hex* env) { env->tick = 0; }

void add_log(Hex* env)
{
	env->log.perf += (env->rewards[0] > 0) ? 1 : 0;
	env->log.score += env->rewards[0];
	env->log.episode_length += env->tick;
	env->log.episode_return += env->rewards[0];
	env->log.n++;
}

// --- Union-Find (Disjoint Set) Logic ---
void uf_init(Hex* env)
{
	for (int i = 0; i < TOTAL_NODES; i++) {
		env->parent[i] = i;
		env->size[i] = 1;
	}
}

int uf_find(Hex* env, int i)
{
	int root = i;
	while (root != env->parent[root]) {
		root = env->parent[root];
	}
	// Path compression
	int curr = i;
	while (curr != root) {
		int nxt = env->parent[curr];
		env->parent[curr] = root;
		curr = nxt;
	}
	return root;
}

void uf_union(Hex* env, int i, int j)
{
	int root_i = uf_find(env, i);
	int root_j = uf_find(env, j);
	if (root_i != root_j) {
		// Union by size
		if (env->size[root_i] < env->size[root_j]) {
			env->parent[root_i] = root_j;
			env->size[root_j] += env->size[root_i];
		} else {
			env->parent[root_j] = root_i;
			env->size[root_i] += env->size[root_j];
		}
	}
}
// --- End Union-Find Logic ---

// Required function
void c_reset(Hex* env)
{
	// set board to empty board
	memset(env->board, 0, sizeof(env->board));
	env->current_player = 0;
	env->tick = 0;
	env->terminals[0] = 0;

	uf_init(env);

	for (int i = 0; i < 2 * TOTAL_CELLS; i++) {
		env->observations[i] = 0;
	}
}

bool invalid_move(int action, const int8_t* board)
{
	if (action < 0 || action >= TOTAL_CELLS) {
		return true; // Out of bounds
	}
	if (board[action] != 0) {
		return true; // Cell already occupied
	}
	return false;
}

int compute_legal_move(Hex* env)
{
	// Naive random move for the environment
	int action;
	do {
		action = rand_r(&env->rng) % TOTAL_CELLS;
	} while (invalid_move(action, env->board));

	return action;
}

int compute_env_move(Hex* env, int player_last_action)
{

	// Play the center if available
	if (env->tick <= 1) {
		int center = (BOARD_SIZE / 2) * BOARD_SIZE + (BOARD_SIZE / 2);
		if (env->board[center] == 0) {
			return center;
		} else {
			return center + 1;
		}
	}

	int best_action = -1;
	int max_c = -1; // ENV (RED) wants to maximize column index (progress to the right).

	// Get the coordinates of the player's last move.
	int r = player_last_action / BOARD_SIZE;
	int c = player_last_action % BOARD_SIZE;

	for (int d = 0; d < 6; d++) {
		int nr = r + dr[d];
		int nc = c + dc[d];
		// Check if the neighbor is on the board
		if (nr >= 0 && nr < BOARD_SIZE && nc >= 0 && nc < BOARD_SIZE) {
			int n_idx = nr * BOARD_SIZE + nc;

			// Check if the neighbor is empty
			if (env->board[n_idx] == 0) {
				// Heuristic: Is this the best blocking move so far?
				// "Best" is defined as making the most progress for us.
				if (nc > max_c) {
					max_c = nc;
					best_action = n_idx;
				}
			}
		}
	}

	// --- FALLBACK ---
	// If best_action is still -1, it means all 6 neighbors were occupied.
	// In this case, we just play random
	if (best_action == -1) {
		return compute_legal_move(env);
	}

	return best_action;
}

// Places a stone, merges components, and returns true if the player won
bool place_stone_and_check_win(Hex* env, int action, int player)
{
	env->board[action] = player;
	int offset = 0;
	if (player == ENV_COLOR) {
		offset = TOTAL_CELLS;
	}
	env->observations[action + offset] = 1;

	int r = action / BOARD_SIZE;
	int c = action % BOARD_SIZE;

	// 1. Connect to adjacent stones of the same color
	for (int i = 0; i < 6; i++) {
		int nr = r + dr[i];
		int nc = c + dc[i];

		if (nr >= 0 && nr < BOARD_SIZE && nc >= 0 && nc < BOARD_SIZE) {
			int n_idx = nr * BOARD_SIZE + nc;
			if (env->board[n_idx] == player) {
				uf_union(env, action, n_idx);
			}
		}
	}

	// 2. Connect to virtual edges and check for a winner
	if (player == PLAYER_COLOR) {
		if (r == 0)
			uf_union(env, action, TOP_NODE);
		if (r == BOARD_SIZE - 1)
			uf_union(env, action, BOTTOM_NODE);

		return uf_find(env, TOP_NODE) == uf_find(env, BOTTOM_NODE);
	} else {
		if (c == 0)
			uf_union(env, action, LEFT_NODE);
		if (c == BOARD_SIZE - 1)
			uf_union(env, action, RIGHT_NODE);

		return uf_find(env, LEFT_NODE) == uf_find(env, RIGHT_NODE);
	}
}

// Required function
void c_step(Hex* env)
{
	env->tick += 1;
	int action = (int)env->actions[0];

	if (invalid_move(action, env->board)) {
		env->rewards[0] = -1;
		env->terminals[0] = 1;
		add_log(env);
		c_reset(env);
		return;
	}

	// Player move and incremental win check
	if (place_stone_and_check_win(env, action, PLAYER_COLOR)) {
		env->rewards[0] = 1;
		env->terminals[0] = 1;

		add_log(env);
		c_reset(env);
		return;
	}
	int env_action;

	if (env->random_opponent) {
		env_action = compute_legal_move(env);

	} else {
		env_action = compute_env_move(env, action);
	}

	if (place_stone_and_check_win(env, env_action, ENV_COLOR)) {
		env->rewards[0] = -1;
		env->terminals[0] = 1;

		add_log(env);
		c_reset(env);
		return;
	}
}

// Required function. Should handle creating the client on first call
void c_render(Hex* env)
{
	int screen_width = 800;
	int screen_height = 600;

	if (!IsWindowReady()) {
		InitWindow(screen_width, screen_height, "PufferLib Hex");
		SetTargetFPS(60);
	}

	if (IsKeyDown(KEY_ESCAPE)) {
		exit(0);
	}

	BeginDrawing();
	ClearBackground((Color) { 6, 24, 24, 255 });

	float radius = 22.0f;
	float sqrt3 = 1.73205f;
	float hex_width = sqrt3 * radius;
	float hex_height = 2.0f * radius;

	float total_width = hex_width * BOARD_SIZE + hex_width * 0.5f * BOARD_SIZE;
	float total_height = hex_height * 0.75f * BOARD_SIZE;

	float start_x = screen_width / 2.0f - total_width / 2.0f + hex_width / 2.0f;
	float start_y = screen_height / 2.0f - total_height / 2.0f + hex_height / 2.0f;

	// Draw borders to show player targets (Blue connects Top/Bottom, Red connects Left/Right)
	for (int r = 0; r < BOARD_SIZE; r++) {
		float left_x = start_x + (0 + r * 0.5f) * hex_width - hex_width * 0.8f;
		float right_x = start_x + (BOARD_SIZE - 1 + r * 0.5f) * hex_width + hex_width * 0.8f;
		float cy = start_y + r * hex_height * 0.75f;
		DrawCircle(left_x, cy, radius * 0.3f, RED);
		DrawCircle(right_x, cy, radius * 0.3f, RED);
	}

	for (int c = 0; c < BOARD_SIZE; c++) {
		float cx_top = start_x + (c + 0 * 0.5f) * hex_width;
		float cy_top = start_y + 0 * hex_height * 0.75f - hex_height * 0.6f;
		float cx_bot = start_x + (c + (BOARD_SIZE - 1) * 0.5f) * hex_width;
		float cy_bot = start_y + (BOARD_SIZE - 1) * hex_height * 0.75f + hex_height * 0.6f;
		DrawCircle(cx_top, cy_top, radius * 0.3f, BLUE);
		DrawCircle(cx_bot, cy_bot, radius * 0.3f, BLUE);
	}

	for (int r = 0; r < BOARD_SIZE; r++) {
		for (int c = 0; c < BOARD_SIZE; c++) {
			int idx = r * BOARD_SIZE + c;
			int owner = env->board[idx];

			Color color = DARKGRAY;
			if (owner == PLAYER_COLOR)
				color = BLUE;
			else if (owner == ENV_COLOR)
				color = RED;

			float cx = start_x + (c + r * 0.5f) * hex_width;
			float cy = start_y + r * hex_height * 0.75f;

			DrawPoly((Vector2) { cx, cy }, 6, radius - 1.0f, 30.0f, color);
			DrawPolyLines((Vector2) { cx, cy }, 6, radius - 1.0f, 30.0f, BLACK);
		}
	}

	EndDrawing();
}

// Required function. Should clean up anything you allocated
// Do not free env->observations, actions, rewards, terminals
void c_close(Hex* env)
{
	if (IsWindowReady()) {
		CloseWindow();
	}
}