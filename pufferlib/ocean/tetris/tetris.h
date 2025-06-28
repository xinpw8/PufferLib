#include "raylib.h"
#include "tetrominoes.h"
#include <assert.h>
#include <limits.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#define min(a, b) (((a) < (b)) ? (a) : (b))
#define max(a, b) (((a) > (b)) ? (a) : (b))

#define HALF_LINEWIDTH 1
#define SQUARE_SIZE 32

#define ACTION_NO_OP 0
#define ACTION_LEFT 1
#define ACTION_RIGHT 2
#define ACTION_ROTATE 3
#define ACTION_SOFT_DROP 4
#define ACTION_HARD_DROP 5
#define ACTION_HOLD 6

#define TICKS_FALL 4 // how many ticks before the tetromino naturally falls down of one square
#define MAX_TICKS 10000
#define PERSONAL_BEST 12565
#define SCORE_SOFT_DROP 1
#define REWARD_SOFT_DROP 0.01f
#define SCORE_HARD_DROP 2
#define REWARD_HARD_DROP 0.02f
#define REWARD_INVALID_ACTION 0.0f

const int SCORE_COMBO[5] = {0, 100, 300, 500, 1000};
const float REWARD_COMBO[5] = {0, 0.1, 0.3, 0.5, 1.0};

typedef struct Log {
	float perf;
	float score;
	float ep_length;
	float ep_return;
	float lines_deleted;
	float avg_combo;
	float atn_frac_soft_drop;
	float atn_frac_hard_drop;
	float atn_frac_rotate;
	float n;
} Log;

typedef struct Client {
	int total_cols;
	int total_rows;
	int ui_rows;
	int deck_rows;
	int preview_target_rotation;
	int preview_target_col;
} Client;

typedef struct Tetris {
	Client *client;
	Log log;
	float *observations;
	int *actions;
	float *rewards;
	unsigned char *terminals;

	int n_rows;
	int n_cols;
	int deck_size;
	int *grid;
	int tick;
	int tick_fall;
	int score;
	int can_swap;

	int *tetromino_deck;
	int hold_tetromino;
	int cur_position_in_deck;
	int cur_tetromino;
	int cur_tetromino_row;
	int cur_tetromino_col;
	int cur_tetromino_rot;

	float ep_return;
	int lines_deleted;
	int count_combos;
	int atn_count_hard_drop;
	int atn_count_soft_drop;
	int atn_count_rotate;
} Tetris;

void init(Tetris *env) {
	env->grid = (int *)calloc(env->n_rows * env->n_cols, sizeof(int));
	env->tetromino_deck = calloc(env->deck_size, sizeof(int));
}

void allocate(Tetris *env) {
	init(env);
	env->observations = (float *)calloc(
	    env->n_cols * env->n_rows + 6 + NUM_TETROMINOES * env->deck_size + NUM_TETROMINOES, sizeof(float));
	env->actions = (int *)calloc(1, sizeof(int));
	env->rewards = (float *)calloc(1, sizeof(float));
	env->terminals = (unsigned char *)calloc(1, sizeof(unsigned char));
}

void c_close(Tetris *env) {
	free(env->grid);
	free(env->tetromino_deck);
}

void free_allocated(Tetris *env) {
	free(env->actions);
	free(env->observations);
	free(env->terminals);
	free(env->rewards);
	c_close(env);
}

void add_log(Tetris *env) {
	env->log.score += env->score;
	env->log.perf += env->score / ((float)PERSONAL_BEST);
	env->log.ep_length += env->tick;
	env->log.ep_return += env->ep_return;
	env->log.lines_deleted += env->lines_deleted;
	env->log.avg_combo += env->count_combos > 0 ? ((float)env->lines_deleted) / ((float)env->count_combos) : 1.0f;
	env->log.atn_frac_hard_drop += env->atn_count_hard_drop / ((float)env->tick);
	env->log.atn_frac_soft_drop += env->atn_count_soft_drop / ((float)env->tick);
	env->log.atn_frac_rotate += env->atn_count_rotate / ((float)env->tick);
	env->log.n += 1;
}

void compute_observations(Tetris *env) {
	memset(env->observations, 0.0,
	       (env->n_cols * env->n_rows + 6 + NUM_TETROMINOES * env->deck_size + NUM_TETROMINOES) * sizeof(float));

	// content of the grid: 1st channel is the grid, 2nd channel is the
	for (int i = 0; i < env->n_cols * env->n_rows; i++) {
		env->observations[i] = env->grid[i] > 0;
	}

	for (int r = 0; r < SIZE; r++) {
		for (int c = 0; c < SIZE; c++) {
			if (TETROMINOES[env->cur_tetromino][env->cur_tetromino_rot][r][c] == 1) {
				env->observations[(env->cur_tetromino_row + r) * env->n_cols + c + env->cur_tetromino_col] = 2;
			}
		}
	}
	int offset = env->n_cols * env->n_rows;
	env->observations[offset] = env->tick / ((float)MAX_TICKS);
	env->observations[offset + 1] = env->tick_fall / ((float)TICKS_FALL);
	env->observations[offset + 2] = env->cur_tetromino_row / ((float)env->n_rows);
	env->observations[offset + 3] = env->cur_tetromino_col / ((float)env->n_cols);
	env->observations[offset + 4] = env->cur_tetromino_rot;
	env->observations[offset + 5] = env->can_swap;

	// deck, one hot endoded
	int tetromino_id;
	for (int j = 0; j < env->deck_size; j++) {
		tetromino_id = env->tetromino_deck[(env->cur_position_in_deck + j) % env->deck_size];
		env->observations[offset + 4 + j * NUM_TETROMINOES + tetromino_id] = 1;
	}

	// hold, one hot endoded
	if (env->hold_tetromino > -1) {
		env->observations[offset + 4 + env->deck_size * NUM_TETROMINOES + env->hold_tetromino] = 1;
	}
}

void restore_grid(Tetris *env) { memset(env->grid, 0, env->n_rows * env->n_cols * sizeof(int)); }

void initialize_deck(Tetris *env) {
	for (int i = 0; i < env->deck_size; i++) {
		env->tetromino_deck[i] = rand() % NUM_TETROMINOES;
	}
	env->cur_position_in_deck = 0;
	env->cur_tetromino = env->tetromino_deck[env->cur_position_in_deck];
}

void spawn_new_tetromino(Tetris *env) {
	env->tetromino_deck[env->cur_position_in_deck] = rand() % NUM_TETROMINOES;
	env->cur_position_in_deck = (env->cur_position_in_deck + 1) % env->deck_size;
	env->cur_tetromino = env->tetromino_deck[env->cur_position_in_deck];
	env->cur_tetromino_rot = 0;
	env->cur_tetromino_col = env->n_cols / 2;
	env->cur_tetromino_row = 0;
	env->tick_fall = 0;
}

bool can_spawn_new_tetromino(Tetris *env) {
	int next_tetromino = env->tetromino_deck[(env->cur_position_in_deck + 1) % env->deck_size];
	for (int c = 0; c < TETROMINOES_FILLS_COL[next_tetromino][0]; c++) {
		for (int r = 0; r < TETROMINOES_FILLS_ROW[next_tetromino][0]; r++) {
			if ((env->grid[r * env->n_cols + c + env->n_cols / 2] > 0) && (TETROMINOES[next_tetromino][0][r][c] == 1)) {
				return false;
			}
		}
	}
	return true;
}

bool can_soft_drop(Tetris *env) {
	if (env->cur_tetromino_row == (env->n_rows - TETROMINOES_FILLS_ROW[env->cur_tetromino][env->cur_tetromino_rot])) {
		return false;
	}
	for (int c = 0; c < TETROMINOES_FILLS_COL[env->cur_tetromino][env->cur_tetromino_rot]; c++) {
		for (int r = 0; r < TETROMINOES_FILLS_ROW[env->cur_tetromino][env->cur_tetromino_rot]; r++) {
			if ((env->grid[(r + env->cur_tetromino_row + 1) * env->n_cols + c + env->cur_tetromino_col] > 0) &&
			    (TETROMINOES[env->cur_tetromino][env->cur_tetromino_rot][r][c] == 1)) {
				return false;
			}
		}
	}
	return true;
}

bool can_go_left(Tetris *env) {
	if (env->cur_tetromino_col == 0) {
		return false;
	}
	for (int c = 0; c < TETROMINOES_FILLS_COL[env->cur_tetromino][env->cur_tetromino_rot]; c++) {
		for (int r = 0; r < TETROMINOES_FILLS_ROW[env->cur_tetromino][env->cur_tetromino_rot]; r++) {
			if ((env->grid[(r + env->cur_tetromino_row) * env->n_cols + c + env->cur_tetromino_col - 1] > 0) &&
			    (TETROMINOES[env->cur_tetromino][env->cur_tetromino_rot][r][c] == 1)) {
				return false;
			}
		}
	}
	return true;
}

bool can_go_right(Tetris *env) {
	if (env->cur_tetromino_col == (env->n_cols - TETROMINOES_FILLS_COL[env->cur_tetromino][env->cur_tetromino_rot])) {
		return false;
	}
	for (int c = 0; c < TETROMINOES_FILLS_COL[env->cur_tetromino][env->cur_tetromino_rot]; c++) {
		for (int r = 0; r < TETROMINOES_FILLS_ROW[env->cur_tetromino][env->cur_tetromino_rot]; r++) {
			if ((env->grid[(r + env->cur_tetromino_row) * env->n_cols + c + env->cur_tetromino_col + 1] > 0) &&
			    (TETROMINOES[env->cur_tetromino][env->cur_tetromino_rot][r][c] == 1)) {
				return false;
			}
		}
	}
	return true;
}

bool can_hold(Tetris *env) {
	if (env->can_swap == 0) {
		return false;
	}
	if (env->hold_tetromino == -1) {
		return true;
	}
	for (int c = 0; c < TETROMINOES_FILLS_COL[env->hold_tetromino][env->cur_tetromino_rot]; c++) {
		for (int r = 0; r < TETROMINOES_FILLS_ROW[env->hold_tetromino][env->cur_tetromino_rot]; r++) {
			if ((env->grid[(r + env->cur_tetromino_row) * env->n_cols + c + env->cur_tetromino_col + 1] > 0) &&
			    (TETROMINOES[env->hold_tetromino][env->cur_tetromino_rot][r][c] == 1)) {
				return false;
			}
		}
	}
	return true;
}

bool can_rotate(Tetris *env) {
	int next_rot = (env->cur_tetromino_rot + 1) % NUM_ROTATIONS;
	if (env->cur_tetromino_col > (env->n_cols - TETROMINOES_FILLS_COL[env->cur_tetromino][next_rot])) {
		return false;
	}
	if (env->cur_tetromino_row > (env->n_rows - TETROMINOES_FILLS_ROW[env->cur_tetromino][next_rot])) {
		return false;
	}
	for (int c = 0; c < TETROMINOES_FILLS_COL[env->cur_tetromino][next_rot]; c++) {
		for (int r = 0; r < TETROMINOES_FILLS_ROW[env->cur_tetromino][next_rot]; r++) {
			if ((env->grid[(r + env->cur_tetromino_row) * env->n_cols + c + env->cur_tetromino_col] > 0) &&
			    (TETROMINOES[env->cur_tetromino][next_rot][r][c] == 1)) {
				return false;
			}
		}
	}
	return true;
}

bool is_full_row(Tetris *env, int row) {
	for (int c = 0; c < env->n_cols; c++) {
		if (env->grid[row * env->n_cols + c] == 0) {
			return false;
		}
	}
	return true;
}

void clear_row(Tetris *env, int row) {
	for (int r = row; r > 0; r--) {
		for (int c = 0; c < env->n_cols; c++) {
			env->grid[r * env->n_cols + c] = env->grid[(r - 1) * env->n_cols + c];
		}
	}
	for (int c = 0; c < env->n_cols; c++) {
		env->grid[c] = 0;
	}
}

void c_reset(Tetris *env) {
	env->score = 0;
	env->hold_tetromino = -1;
	env->tick = 0;
	env->tick_fall = 0;
	env->can_swap = 1;

	env->ep_return = 0.0;
	env->count_combos = 0;
	env->lines_deleted = 0;
	env->atn_count_hard_drop = 0;
	env->atn_count_soft_drop = 0;
	env->atn_count_rotate = 0;

	restore_grid(env);
	initialize_deck(env);
	spawn_new_tetromino(env);
	compute_observations(env);
}

void place_tetromino(Tetris *env) {
	int row_to_check = env->cur_tetromino_row + TETROMINOES_FILLS_ROW[env->cur_tetromino][env->cur_tetromino_rot] - 1;
	int lines_deleted = 0;
	env->can_swap = 1;

	for (int c = 0; c < TETROMINOES_FILLS_COL[env->cur_tetromino][env->cur_tetromino_rot];
	     c++) { // Fill the main grid with the tetromino
		for (int r = 0; r < TETROMINOES_FILLS_ROW[env->cur_tetromino][env->cur_tetromino_rot]; r++) {
			if (TETROMINOES[env->cur_tetromino][env->cur_tetromino_rot][r][c] == 1) {
				env->grid[(r + env->cur_tetromino_row) * env->n_cols + c + env->cur_tetromino_col] =
				    env->cur_tetromino + 1;
			}
		}
	}
	for (int r = 0; r < TETROMINOES_FILLS_ROW[env->cur_tetromino][env->cur_tetromino_rot];
	     r++) { // Proceed to delete the complete rows
		if (is_full_row(env, row_to_check)) {
			clear_row(env, row_to_check);
			lines_deleted += 1;
		} else {
			row_to_check -= 1;
		}
	}
	if (lines_deleted > 0) {
		env->count_combos += 1;
		env->lines_deleted += lines_deleted;
		env->score += SCORE_COMBO[lines_deleted];
		env->rewards[0] += REWARD_COMBO[lines_deleted];
		env->ep_return += REWARD_COMBO[lines_deleted];
	}
	if ((!can_spawn_new_tetromino(env)) || (env->tick >= MAX_TICKS)) {
		env->terminals[0] = 1;
		add_log(env);
		c_reset(env);
	} else {
		spawn_new_tetromino(env);
	}
}

void c_step(Tetris *env) {
	env->terminals[0] = 0;
	env->rewards[0] = 0.0;
	env->tick += 1;
	env->tick_fall += 1;
	int action = env->actions[0];

	if (action == ACTION_LEFT) {
		if (can_go_left(env)) {
			env->cur_tetromino_col -= 1;
		} else {
			env->rewards[0] += REWARD_INVALID_ACTION;
			env->ep_return += REWARD_INVALID_ACTION;
			// action = ACTION_HARD_DROP;
		}
	}
	if (action == ACTION_RIGHT) {
		if (can_go_right(env)) {
			env->cur_tetromino_col += 1;
		} else {
			env->rewards[0] += REWARD_INVALID_ACTION;
			env->ep_return += REWARD_INVALID_ACTION;
			// action = ACTION_HARD_DROP;
		}
	}
	if (action == ACTION_ROTATE) {
		env->atn_count_rotate += 1;
		if (can_rotate(env)) {
			env->cur_tetromino_rot = (env->cur_tetromino_rot + 1) % NUM_ROTATIONS;
		} else {
			env->rewards[0] += REWARD_INVALID_ACTION;
			env->ep_return += REWARD_INVALID_ACTION;
			// action = ACTION_HARD_DROP;
		}
	}
	if (action == ACTION_SOFT_DROP) {
		env->atn_count_soft_drop += 1;
		if (can_soft_drop(env)) {
			env->cur_tetromino_row += 1;
			env->score += SCORE_SOFT_DROP;
			env->rewards[0] += REWARD_SOFT_DROP;
			env->ep_return += REWARD_SOFT_DROP;
		} else {
			env->rewards[0] += REWARD_INVALID_ACTION;
			env->ep_return += REWARD_INVALID_ACTION;
			// action = ACTION_HARD_DROP;
		}
	}
	if (action == ACTION_HOLD) {
		if (can_hold(env)) {
			int t1 = env->cur_tetromino;
			int t2 = env->hold_tetromino;
			if (t2 == -1) {
				spawn_new_tetromino(env);
				env->hold_tetromino = t1;
				env->can_swap = 0;
			} else {
				env->cur_tetromino = t2;
				env->tetromino_deck[env->cur_position_in_deck] = t2;
				env->hold_tetromino = t1;
				env->can_swap = 0;
				env->cur_tetromino_rot = 0;
				env->cur_tetromino_col = env->n_cols / 2;
				env->cur_tetromino_row = 0;
				env->tick_fall = 0;
			}
		} else {
			env->rewards[0] += REWARD_INVALID_ACTION;
			env->ep_return += REWARD_INVALID_ACTION;
			// action = ACTION_HARD_DROP;
		}
	}
	if (action == ACTION_HARD_DROP) {
		env->atn_count_hard_drop += 1;
		while (can_soft_drop(env)) {
			env->cur_tetromino_row += 1;
			env->score += SCORE_HARD_DROP;
			env->rewards[0] += REWARD_HARD_DROP;
			env->ep_return += REWARD_HARD_DROP;
		}
		place_tetromino(env);
	}
	if (env->tick_fall == TICKS_FALL) {
		env->tick_fall = 0;
		if (!can_soft_drop(env)) {
			place_tetromino(env);
		} else {
			env->cur_tetromino_row += 1;
		}
	}
	compute_observations(env);
}

Client *make_client(Tetris *env) {
	Client *client = (Client *)calloc(1, sizeof(Client));
	client->ui_rows = 1;
	client->deck_rows = SIZE;
	client->total_rows = 1 + client->ui_rows + 1 + client->deck_rows + 1 + env->n_rows + 1;
	client->total_cols = max(1 + env->n_cols + 1, 1 + 3 * (env->deck_size - 1));
	client->preview_target_col = env->n_cols / 2;
	client->preview_target_rotation = 0;
	InitWindow(SQUARE_SIZE * client->total_cols, SQUARE_SIZE * client->total_rows, "PufferLib Tetris");
	SetTargetFPS(10);
	return client;
}

void close_client(Client *client) {
	CloseWindow();
	free(client);
}

Color BORDER_COLOR = (Color){100, 100, 100, 255};
Color DASH_COLOR = (Color){80, 80, 80, 255};
Color DASH_COLOR_BRIGHT = (Color){150, 150, 150, 255};
Color DASH_COLOR_DARK = (Color){50, 50, 50, 255};

void c_render(Tetris *env) {
	if (env->client == NULL) {
		env->client = make_client(env);
	}
	Client *client = env->client;

	if (IsKeyDown(KEY_ESCAPE)) {
		exit(0);
	}
	if (IsKeyPressed(KEY_TAB)) {
		ToggleFullscreen();
	}

	BeginDrawing();
	ClearBackground(BLACK);
	int x, y;
	Color color;

	// outer grid
	for (int r = 0; r < client->total_rows; r++) {
		for (int c = 0; c < client->total_cols; c++) {
			x = c * SQUARE_SIZE;
			y = r * SQUARE_SIZE;
			if ((c == 0) || (c == client->total_cols - 1) ||
			    ((r >= 1 + client->ui_rows + 1) && (r < 1 + client->ui_rows + 1 + client->deck_rows)) ||
			    ((r >= 1 + client->ui_rows + 1 + client->deck_rows + 1) && (c >= env->n_rows)) || (r == 0) ||
			    (r == 1 + client->ui_rows) || (r == 1 + client->ui_rows + 1 + client->deck_rows) ||
			    (r == client->total_rows - 1)) {
				DrawRectangle(x + HALF_LINEWIDTH, y + HALF_LINEWIDTH, SQUARE_SIZE - 2 * HALF_LINEWIDTH,
				              SQUARE_SIZE - 2 * HALF_LINEWIDTH, BORDER_COLOR);
				DrawRectangle(x - HALF_LINEWIDTH, y - HALF_LINEWIDTH, SQUARE_SIZE, 2 * HALF_LINEWIDTH, DASH_COLOR_DARK);
				DrawRectangle(x - HALF_LINEWIDTH, y + SQUARE_SIZE - HALF_LINEWIDTH, SQUARE_SIZE, 2 * HALF_LINEWIDTH,
				              DASH_COLOR_DARK);
				DrawRectangle(x - HALF_LINEWIDTH, y - HALF_LINEWIDTH, 2 * HALF_LINEWIDTH, SQUARE_SIZE, DASH_COLOR_DARK);
				DrawRectangle(x + SQUARE_SIZE - HALF_LINEWIDTH, y - HALF_LINEWIDTH, 2 * HALF_LINEWIDTH, SQUARE_SIZE,
				              DASH_COLOR_DARK);
			}
		}
	}
	// main grid
	for (int r = 0; r < env->n_rows; r++) {
		for (int c = 0; c < env->n_cols; c++) {
			x = (c + 1) * SQUARE_SIZE;
			y = (1 + client->ui_rows + 1 + client->deck_rows + 1 + r) * SQUARE_SIZE;
			color =
			    (env->grid[r * env->n_cols + c] == 0) ? BLACK : TETROMINOES_COLORS[env->grid[r * env->n_cols + c] - 1];
			DrawRectangle(x + HALF_LINEWIDTH, y + HALF_LINEWIDTH, SQUARE_SIZE - 2 * HALF_LINEWIDTH,
			              SQUARE_SIZE - 2 * HALF_LINEWIDTH, color);
			DrawRectangle(x - HALF_LINEWIDTH, y - HALF_LINEWIDTH, SQUARE_SIZE, 2 * HALF_LINEWIDTH, DASH_COLOR);
			DrawRectangle(x - HALF_LINEWIDTH, y + SQUARE_SIZE - HALF_LINEWIDTH, SQUARE_SIZE, 2 * HALF_LINEWIDTH,
			              DASH_COLOR);
			DrawRectangle(x - HALF_LINEWIDTH, y - HALF_LINEWIDTH, 2 * HALF_LINEWIDTH, SQUARE_SIZE, DASH_COLOR);
			DrawRectangle(x + SQUARE_SIZE - HALF_LINEWIDTH, y - HALF_LINEWIDTH, 2 * HALF_LINEWIDTH, SQUARE_SIZE,
			              DASH_COLOR);
		}
	}

	// current tetromino
	for (int r = 0; r < SIZE; r++) {
		for (int c = 0; c < SIZE; c++) {
			x = (c + env->cur_tetromino_col + 1) * SQUARE_SIZE;
			y = (1 + client->ui_rows + 1 + client->deck_rows + 1 + r + env->cur_tetromino_row) * SQUARE_SIZE;

			if (TETROMINOES[env->cur_tetromino][env->cur_tetromino_rot][r][c] == 1) {
				color = TETROMINOES_COLORS[env->cur_tetromino];
				DrawRectangle(x + HALF_LINEWIDTH, y + HALF_LINEWIDTH, SQUARE_SIZE - 2 * HALF_LINEWIDTH,
				              SQUARE_SIZE - 2 * HALF_LINEWIDTH, color);
				DrawRectangle(x - HALF_LINEWIDTH, y - HALF_LINEWIDTH, SQUARE_SIZE, 2 * HALF_LINEWIDTH, DASH_COLOR);
				DrawRectangle(x - HALF_LINEWIDTH, y + SQUARE_SIZE - HALF_LINEWIDTH, SQUARE_SIZE, 2 * HALF_LINEWIDTH,
				              DASH_COLOR);
				DrawRectangle(x - HALF_LINEWIDTH, y - HALF_LINEWIDTH, 2 * HALF_LINEWIDTH, SQUARE_SIZE, DASH_COLOR);
				DrawRectangle(x + SQUARE_SIZE - HALF_LINEWIDTH, y - HALF_LINEWIDTH, 2 * HALF_LINEWIDTH, SQUARE_SIZE,
				              DASH_COLOR);
			}
		}
	}

	// Deck grid
	int tetromino_id;
	for (int i = 0; i < env->deck_size - 1; i++) {
		tetromino_id = env->tetromino_deck[(env->cur_position_in_deck + 1 + i) % env->deck_size];
		for (int r = 0; r < SIZE; r++) {
			for (int c = 0; c < 2; c++) {
				x = (c + 1 + 3 * i) * SQUARE_SIZE;
				y = (1 + client->ui_rows + 1 + r) * SQUARE_SIZE;
				int r_offset = (SIZE - TETROMINOES_FILLS_ROW[tetromino_id][0]);
				if (r < r_offset) {
					color = BLACK;
				} else {
					color =
					    (TETROMINOES[tetromino_id][0][r - r_offset][c] == 0) ? BLACK : TETROMINOES_COLORS[tetromino_id];
				}
				DrawRectangle(x + HALF_LINEWIDTH, y + HALF_LINEWIDTH, SQUARE_SIZE - 2 * HALF_LINEWIDTH,
				              SQUARE_SIZE - 2 * HALF_LINEWIDTH, color);
				DrawRectangle(x - HALF_LINEWIDTH, y - HALF_LINEWIDTH, SQUARE_SIZE, 2 * HALF_LINEWIDTH,
				              DASH_COLOR_BRIGHT);
				DrawRectangle(x - HALF_LINEWIDTH, y + SQUARE_SIZE - HALF_LINEWIDTH, SQUARE_SIZE, 2 * HALF_LINEWIDTH,
				              DASH_COLOR_BRIGHT);
				DrawRectangle(x - HALF_LINEWIDTH, y - HALF_LINEWIDTH, 2 * HALF_LINEWIDTH, SQUARE_SIZE,
				              DASH_COLOR_BRIGHT);
				DrawRectangle(x + SQUARE_SIZE - HALF_LINEWIDTH, y - HALF_LINEWIDTH, 2 * HALF_LINEWIDTH, SQUARE_SIZE,
				              DASH_COLOR_BRIGHT);
			}
		}
	}

	// hold tetromino
	for (int r = 0; r < SIZE; r++) {
		for (int c = 0; c < 2; c++) {
			x = (client->total_cols - 3 + c) * SQUARE_SIZE;
			y = (1 + client->ui_rows + 1 + r) * SQUARE_SIZE;
			if (env->hold_tetromino > -1) {
				int r_offset = (SIZE - TETROMINOES_FILLS_ROW[env->hold_tetromino][0]);
				if (r < r_offset) {
					color = BLACK;
				} else {
					color = (env->hold_tetromino > -1) && (TETROMINOES[env->hold_tetromino][0][r - r_offset][c] == 0)
					            ? BLACK
					            : TETROMINOES_COLORS[env->hold_tetromino];
				}
			} else {
				color = BLACK;
			}
			DrawRectangle(x + HALF_LINEWIDTH, y + HALF_LINEWIDTH, SQUARE_SIZE - 2 * HALF_LINEWIDTH,
			              SQUARE_SIZE - 2 * HALF_LINEWIDTH, color);
			DrawRectangle(x - HALF_LINEWIDTH, y - HALF_LINEWIDTH, SQUARE_SIZE, 2 * HALF_LINEWIDTH, DASH_COLOR_BRIGHT);
			DrawRectangle(x - HALF_LINEWIDTH, y + SQUARE_SIZE - HALF_LINEWIDTH, SQUARE_SIZE, 2 * HALF_LINEWIDTH,
			              DASH_COLOR_BRIGHT);
			DrawRectangle(x - HALF_LINEWIDTH, y - HALF_LINEWIDTH, 2 * HALF_LINEWIDTH, SQUARE_SIZE, DASH_COLOR_BRIGHT);
			DrawRectangle(x + SQUARE_SIZE - HALF_LINEWIDTH, y - HALF_LINEWIDTH, 2 * HALF_LINEWIDTH, SQUARE_SIZE,
			              DASH_COLOR_BRIGHT);
		}
	}
	// Draw UI
	DrawText(TextFormat("Score: %i", env->score), SQUARE_SIZE + 4, SQUARE_SIZE + 4, 30, (Color){255, 160, 160, 255});
	EndDrawing();
}
