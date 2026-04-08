/* Squared: a sample single-agent grid env.
 * Use this as a tutorial and template for your first env.
 * See the Target env for a slightly more complex example.
 * Star PufferLib on GitHub to support. It really, really helps!
 */

#include "raylib.h"
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define BOARD_SIZE 11
#define TOTAL_CELLS (BOARD_SIZE * BOARD_SIZE)
#define PLAYER_COLOR 1
#define ENV_COLOR -1

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
	float* observations; // Required. You can use any obs type, but make sure it
						 // matches in Python!
	float* actions; // Required
	float* rewards; // Required
	float* terminals; // Required
	int num_agents;
	int tick;
	int current_player;
	int8_t board[TOTAL_CELLS];

	unsigned int rng;
} Hex;

void allocate_chex(Hex* env)
{
	env->observations = (float*)calloc(TOTAL_CELLS, sizeof(float));
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

void init(Hex* env) {
    env->tick = 0;
}

void add_log(Hex* env)
{
	env->log.perf += (env->rewards[0] > 0) ? 1 : 0;
	env->log.score += env->rewards[0];
	env->log.episode_length += env->tick;
	env->log.episode_return += env->rewards[0];
	env->log.n++;
}

// Required function
void c_reset(Hex* env)
{

	// set board to empty board
	memset(env->board, 0, sizeof(env->board));
	env->current_player = 0;
	env->tick = 0;
	env->terminals[0] = 0;
	int target_idx = 0; // Deterministic for testing

	for (int i = 0; i < TOTAL_CELLS; i++) {
		env->observations[i] = (float)env->board[i];
	}
}
// A recursive DFS function to find a path to the opposite edge
bool dfs(const int8_t* board, bool* visited, int player, int r, int c)
{
	// Mark the current hex as visited
	visited[r * BOARD_SIZE + c] = true;

	// Did we reach the target edge?
	if (player == PLAYER_COLOR && r == BOARD_SIZE - 1)
		return true; // Reached bottom
	if (player == ENV_COLOR && c == BOARD_SIZE - 1)
		return true; // Reached right

	// Check all 6 neighbors
	for (int i = 0; i < 6; i++) {
		int nr = r + dr[i];
		int nc = c + dc[i];

		// Is the neighbor inside the board?
		if (nr >= 0 && nr < BOARD_SIZE && nc >= 0 && nc < BOARD_SIZE) {
			int n_idx = nr * BOARD_SIZE + nc;

			// If the neighbor belongs to the player and hasn't been visited
			if (!visited[n_idx] && board[n_idx] == player) {
				// Recursively search from that neighbor
				if (dfs(board, visited, player, nr, nc)) {
					return true;
				}
			}
		}
	}

	return false; // No path found from this cell
}


bool is_player_winner(const int8_t* board, int player)
{
	bool visited[TOTAL_CELLS];

	memset(visited, 0, sizeof(visited)); // Clear visited array

	if (player == PLAYER_COLOR) {
		for (int c = 0; c < BOARD_SIZE; c++) {
			if (board[c] == PLAYER_COLOR && !visited[0 * BOARD_SIZE + c]) {
				if (dfs(board, visited, player, 0, c)) {
					return true;
				}
			}
		}
	} else {

		for (int r = 0; r < BOARD_SIZE; r++) {
			if (board[r * BOARD_SIZE] == ENV_COLOR && !visited[r * BOARD_SIZE]) {
				if (dfs(board, visited, player, r, 0)) {
					return true;
				}
			}
		}
	}

	return false;
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

int compute_env_move(Hex* env)
{
	// Naive random move for the environment
	int action;
	do {
		action = rand_r(&env->rng) % TOTAL_CELLS;
	} while (invalid_move(action, env->board));



	return action;
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

	int r = action / BOARD_SIZE;
	int c = action % BOARD_SIZE;

	env->board[action] = PLAYER_COLOR;
	env->observations[action] = (float)PLAYER_COLOR;

    bool is_winner = is_player_winner(env->board, PLAYER_COLOR);

    if (is_winner) {
        env->rewards[0] = 1;
        env->terminals[0] = 1;
		add_log(env);
		c_reset(env);
        return;
    }

	action = compute_env_move(env);

	env->board[action] = ENV_COLOR;
	env->observations[action] = (float)ENV_COLOR;

    is_winner = is_player_winner(env->board, ENV_COLOR);

	if (is_winner) {
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
	ClearBackground((Color){6, 24, 24, 255});

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
			if (owner == PLAYER_COLOR) color = BLUE;
			else if (owner == ENV_COLOR) color = RED;

			float cx = start_x + (c + r * 0.5f) * hex_width;
			float cy = start_y + r * hex_height * 0.75f;

			DrawPoly((Vector2){cx, cy}, 6, radius - 1.0f, 30.0f, color);
			DrawPolyLines((Vector2){cx, cy}, 6, radius - 1.0f, 30.0f, BLACK);
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
