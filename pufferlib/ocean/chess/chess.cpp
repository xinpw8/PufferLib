// chess.cpp - Graphical Chess Evaluation using Raylib
#include <algorithm>
#include <chrono>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <math.h>
#include <random>
#include <sstream>
#include <string>
#include <time.h>
#include <unistd.h>
#include <unordered_map>
#include <utility>
#include <vector>

// POSIX headers for directory operations and iostream for cerr
#include <cerrno>
#include <dirent.h>
#include <iostream>
#include <sys/stat.h>

// Conditionally enable the replay feature if the json library is available
#if defined(__has_include)
#if __has_include(<nlohmann/json.hpp>)
#include <nlohmann/json.hpp>
#define PUFFER_REPLAY_ENABLED 1
#else
#define PUFFER_REPLAY_ENABLED 0
#endif
#else
#define PUFFER_REPLAY_ENABLED 0
#endif

// PufferLib C headers for the neural network
#include "../../extensions/puffernet.h"

#ifdef __cplusplus
extern "C" {
#endif
#include "chess.h" // CONTAINS ALL ENV LOGIC!! NO CHESS LOGIC IN chess.cpp!!!
// We train on chess.h and we eval on the identical chess.h that we trained on!!
#include "chess_action_mapping.h"
#include "raylib.h"
#ifdef __cplusplus
}
#endif

// Preserve raylib color constants
static const Color RL_WHITE = WHITE;
static const Color RL_BLACK = BLACK;
static const Color RL_RAYWHITE = RAYWHITE;
static const Color RL_DARKGRAY = DARKGRAY;
static const Color RL_RED = RED;
static const Color RL_BLUE = BLUE;
static const Color RL_ORANGE = ORANGE;
static const Color RL_BEIGE = BEIGE;
static const Color RL_BROWN = BROWN;
static const Color RL_LIGHTGRAY = LIGHTGRAY;
static const Color RL_DARKGREEN = DARKGREEN;
static const Color RL_DARKBLUE = DARKBLUE;

#ifdef WHITE
#undef WHITE
#endif
#ifdef BLACK
#undef BLACK
#endif

// UI Move struct
typedef struct {
  Square from;
  Square to;
  PieceType promotion;
} UIMove;

namespace chess {

// Game outcome structure
struct GameOutcome {
  bool game_ended = false;
  bool white_won = false;
  bool black_won = false;
  bool is_draw = false;
  std::string draw_reason = "";
};
GameOutcome last_game_outcome;

// ChessNet definition
typedef struct ChessNet ChessNet;
struct ChessNet {
  int num_agents;
  Linear *board_enc1, *board_enc2, *combiner, *policy_head, *value_head1,
      *value_head2;
  ReLU *board_relu1, *board_relu2, *comb_relu, *value_relu;
  LSTM *lstm;
  Multidiscrete *md;
};

#define CHESS_NUM_WEIGHTS 2646339

static inline void mask_logits(float *logits, const float *legal, int size) {
  for (int i = 0; i < size; ++i) {
    if (legal[i] < 0.5f)
      logits[i] = -1e9f;
  }
}

static ChessNet *init_chessnet(Weights *weights, int num_agents) {
  ChessNet *net = (ChessNet *)calloc(1, sizeof(ChessNet));
  net->num_agents = num_agents;
  net->board_enc1 = make_linear(weights, num_agents, 1344, 512);
  net->board_relu1 = make_relu(num_agents, 512);
  net->board_enc2 = make_linear(weights, num_agents, 512, 256);
  net->board_relu2 = make_relu(num_agents, 256);
  net->combiner = make_linear(weights, num_agents, 256, 256);
  net->comb_relu = make_relu(num_agents, 256);
  net->lstm = make_lstm(weights, num_agents, 256, 256);
  net->policy_head = make_linear(weights, num_agents, 256, 1968);
  net->value_head1 = make_linear(weights, num_agents, 256, 128);
  net->value_relu = make_relu(num_agents, 128);
  net->value_head2 = make_linear(weights, num_agents, 128, 1);
  int logit_sizes[1] = {1968};
  net->md = make_multidiscrete(num_agents, logit_sizes, 1);
  return net;
}

static void free_chessnet(ChessNet *net) {
  free(net->board_enc1);
  free(net->board_relu1);
  free(net->board_enc2);
  free(net->board_relu2);
  free(net->combiner);
  free(net->comb_relu);
  free(net->lstm);
  free(net->policy_head);
  free(net->value_head1);
  free(net->value_relu);
  free(net->value_head2);
  free(net->md);
  free(net);
}

static void forward_chessnet(ChessNet *net, float *observations, int *actions) {
  const float *board_obs = observations;
  const float *legal_mask = observations + 1344; // This is 1968-dimensional from chess.h
  
  linear(net->board_enc1, board_obs);
  relu(net->board_relu1, net->board_enc1->output);
  linear(net->board_enc2, net->board_relu1->output);
  relu(net->board_relu2, net->board_enc2->output);
  linear(net->combiner, net->board_relu2->output);
  relu(net->comb_relu, net->combiner->output);
  lstm(net->lstm, net->comb_relu->output);
  linear(net->policy_head, net->lstm->state_h);
  
  // Use the 1968-dimensional legal mask directly
  mask_logits(net->policy_head->output, legal_mask, TOTAL_CHESS_ACTIONS);
  softmax_multidiscrete(net->md, net->policy_head->output, actions);
  
  // Ensure output action is within valid range
  if (actions[0] >= TOTAL_CHESS_ACTIONS) {
    printf("[WARNING] Neural network selected invalid action %d, clamping to 0\n", actions[0]);
    actions[0] = 0;
  }
  
  linear(net->value_head1, net->lstm->state_h);
  relu(net->value_relu, net->value_head1->output);
  linear(net->value_head2, net->value_relu->output);
}

// Chess piece textures
typedef struct {
  Texture2D wking, wqueen, wrook, wbishop, wknight, wpawn;
  Texture2D bking, bqueen, brook, bbishop, bknight, bpawn;
} ChessPieceTextures;

static ChessPieceTextures load_piece_textures() {
  ChessPieceTextures textures = {0};
  
  printf("Loading chess piece textures...\n");
  
  textures.wking = LoadTexture("resources/chess/wking.png");
  if (textures.wking.id == 0) printf("Warning: Failed to load wking.png\n");
  
  textures.wqueen = LoadTexture("resources/chess/wqueen.png");
  if (textures.wqueen.id == 0) printf("Warning: Failed to load wqueen.png\n");
  
  textures.wrook = LoadTexture("resources/chess/wrook.png");
  if (textures.wrook.id == 0) printf("Warning: Failed to load wrook.png\n");
  
  textures.wbishop = LoadTexture("resources/chess/wbishop.png");
  if (textures.wbishop.id == 0) printf("Warning: Failed to load wbishop.png\n");
  
  textures.wknight = LoadTexture("resources/chess/wknight.png");
  if (textures.wknight.id == 0) printf("Warning: Failed to load wknight.png\n");
  
  textures.wpawn = LoadTexture("resources/chess/wpawn.png");
  if (textures.wpawn.id == 0) printf("Warning: Failed to load wpawn.png\n");
  
  textures.bking = LoadTexture("resources/chess/bking.png");
  if (textures.bking.id == 0) printf("Warning: Failed to load bking.png\n");
  
  textures.bqueen = LoadTexture("resources/chess/bqueen.png");
  if (textures.bqueen.id == 0) printf("Warning: Failed to load bqueen.png\n");
  
  textures.brook = LoadTexture("resources/chess/brook.png");
  if (textures.brook.id == 0) printf("Warning: Failed to load brook.png\n");
  
  textures.bbishop = LoadTexture("resources/chess/bbishop.png");
  if (textures.bbishop.id == 0) printf("Warning: Failed to load bbishop.png\n");
  
  textures.bknight = LoadTexture("resources/chess/bknight.png");
  if (textures.bknight.id == 0) printf("Warning: Failed to load bknight.png\n");
  
  textures.bpawn = LoadTexture("resources/chess/bpawn.png");
  if (textures.bpawn.id == 0) printf("Warning: Failed to load bpawn.png\n");
  
  printf("Texture loading completed.\n");
  return textures;
}

static void unload_piece_textures(ChessPieceTextures *textures) {
  // Let Raylib's CloseWindow() handle texture cleanup automatically
  // Manual texture unloading can cause double-free issues
  (void)textures; // Suppress unused parameter warning
}

static Texture2D get_piece_texture(const ChessPieceTextures *textures,
                                   PieceColor color, PieceType type) {
  Texture2D empty_texture = {0};
  if (type == EMPTY)
    return empty_texture;
  if (color == C_WHITE) {
    switch (type) {
    case KING:
      return textures->wking;
    case QUEEN:
      return textures->wqueen;
    case ROOK:
      return textures->wrook;
    case BISHOP:
      return textures->wbishop;
    case KNIGHT:
      return textures->wknight;
    case PAWN:
      return textures->wpawn;
    default:
      return empty_texture;
    }
  } else {
    switch (type) {
    case KING:
      return textures->bking;
    case QUEEN:
      return textures->bqueen;
    case ROOK:
      return textures->brook;
    case BISHOP:
      return textures->bbishop;
    case KNIGHT:
      return textures->bknight;
    case PAWN:
      return textures->bpawn;
    default:
      return empty_texture;
    }
  }
}

// UI constants and state
const int BOARD_SIZE = 512, SQUARE_SIZE = BOARD_SIZE / 8, BOARD_OFFSET_X = 50,
          BOARD_OFFSET_Y = 70;
const int WINDOW_WIDTH = 900, WINDOW_HEIGHT = 700;
bool game_paused = false;
std::vector<std::string> game_moves;
bool show_promotion_selection = false;
int promotion_from_x = -1, promotion_from_y = -1, promotion_to_x = -1,
    promotion_to_y = -1;
PieceType selected_promotion = QUEEN;
CChess *global_env_ptr = nullptr;

// Game mode definitions
enum GameMode {
  GM_PLAYER_STOCKFISH,
  GM_PLAYER_RANDOM,
  GM_AGENT_STOCKFISH,
  GM_AGENT_AGENT,
  GM_AGENT_RANDOM,
  GM_RANDOM_RANDOM,
  GM_RANDOM_AGENT,
  GM_GAME_REPLAY,
  GM_COUNT
};
const char *GAME_MODE_NAMES[] = {"Player vs Stockfish", "Player vs Random",
                                 "Agent vs Stockfish",  "Agent vs Agent",
                                 "Agent vs Random",     "Random vs Random",
                                 "Random vs Agent",     "Game Replay"};

#if PUFFER_REPLAY_ENABLED
// Game Replay related variables and structures
GameLogger *global_game_logger = nullptr;
std::vector<std::string> available_games;
int selected_game_index = 0;
bool show_game_list = false;
bool replay_mode_active = false;
struct GameMove {
  int move_number;
  int action_id;
  std::string algebraic_notation;
};
struct GameLogEntry {
  std::string filename, timestamp, outcome, draw_reason;
  int total_moves;
  std::vector<GameMove> moves;
};
struct GameReplay {
  bool is_active = false;
  const GameLogEntry *current_game = nullptr;
  int current_move_index = 0;
  bool start_replay(const GameLogEntry *game) {
    if (!game)
      return false;
    current_game = game;
    current_move_index = 0;
    is_active = true;
    return true;
  }
  bool next_move() {
    if (!is_active || !current_game ||
        current_move_index >= current_game->total_moves - 1)
      return false;
    current_move_index++;
    return true;
  }
  bool prev_move() {
    if (!is_active || !current_game || current_move_index <= 0)
      return false;
    current_move_index--;
    return true;
  }
  bool jump_to_move(int idx) {
    if (!is_active || !current_game || idx < 0 ||
        idx >= current_game->total_moves)
      return false;
    current_move_index = idx;
    return true;
  }
  const GameMove *get_current_move() const {
    if (!is_active || !current_game ||
        current_move_index >= (int)current_game->moves.size())
      return nullptr;
    return ¤t_game->moves[current_move_index];
  }
};
GameReplay current_replay;

// JSON serialization for GameLogEntry
void to_json(nlohmann::json &j, const GameMove &m) {
  j = nlohmann::json{{"move_number", m.move_number},
                     {"action_id", m.action_id},
                     {"algebraic_notation", m.algebraic_notation}};
}
void from_json(const nlohmann::json &j, GameMove &m) {
  j.at("move_number").get_to(m.move_number);
  j.at("action_id").get_to(m.action_id);
  j.at("algebraic_notation").get_to(m.algebraic_notation);
}
void to_json(nlohmann::json &j, const GameLogEntry &e) {
  j = nlohmann::json{
      {"filename", e.filename},       {"timestamp", e.timestamp},
      {"outcome", e.outcome},         {"draw_reason", e.draw_reason},
      {"total_moves", e.total_moves}, {"moves", e.moves}};
}
void from_json(const nlohmann::json &j, GameLogEntry &e) {
  j.at("filename").get_to(e.filename);
  j.at("timestamp").get_to(e.timestamp);
  j.at("outcome").get_to(e.outcome);
  if (j.contains("draw_reason"))
    j.at("draw_reason").get_to(e.draw_reason);
  j.at("total_moves").get_to(e.total_moves);
  j.at("moves").get_to(e.moves);
}

class GameLogger {
private:
  std::vector<GameLogEntry> games;
  std::string log_directory;

public:
  GameLogger(const std::string &directory) : log_directory(directory) {
    struct stat st;
    if (stat(log_directory.c_str(), &st) != 0) {
      std::string cmd = "mkdir -p " + log_directory;
      system(cmd.c_str());
    }
  }
  void load_games_from_directory() {
    games.clear();
    DIR *dir;
    if ((dir = opendir(log_directory.c_str())) != NULL) {
      struct dirent *ent;
      while ((ent = readdir(dir)) != NULL) {
        std::string filename(ent->d_name);
        if (filename.length() > 5 &&
            filename.substr(filename.length() - 5) == ".json") {
          std::string full_path = log_directory + "/" + filename;
          try {
            std::ifstream file(full_path);
            nlohmann::json j;
            file >> j;
            games.push_back(j.get<GameLogEntry>());
          } catch (const std::exception &e) {
            std::cerr << "Error parsing " << full_path << ": " << e.what()
                      << std::endl;
          }
        }
      }
      closedir(dir);
    }
  }
  const std::vector<GameLogEntry> &get_games() const { return games; }
};
#endif // PUFFER_REPLAY_ENABLED

// Session statistics
struct PlayerStats {
  int wins = 0, losses = 0, draws = 0, games = 0;
  float win_rate() const { return games > 0 ? (float)wins / games : 0.0f; }
  void add_win() {
    wins++;
    games++;
  }
  void add_loss() {
    losses++;
    games++;
  }
  void add_draw() {
    draws++;
    games++;
  }
  void reset() { wins = losses = draws = games = 0; }
};
struct SessionStats {
  int total_games = 0, total_wins = 0, total_losses = 0, total_draws = 0;
  PlayerStats agent_stats, human_stats, white_stats, black_stats;
  void reset() {
    total_games = total_wins = total_losses = total_draws = 0;
    agent_stats.reset();
    human_stats.reset();
    white_stats.reset();
    black_stats.reset();
  }
  void print_summary() const {
    printf("=== Session Statistics Summary ===\nTotal Games: %d\n",
           total_games);
    printf("White Wins: %d, Black Wins: %d, Draws: %d\n", total_wins,
           total_losses, total_draws);
    if (agent_stats.games > 0)
      printf("Agent: %.1f%% win rate (%d/%d/%d)\n",
             agent_stats.win_rate() * 100, agent_stats.wins, agent_stats.losses,
             agent_stats.draws);
    if (human_stats.games > 0)
      printf("Human: %.1f%% win rate (%d/%d/%d)\n",
             human_stats.win_rate() * 100, human_stats.wins, human_stats.losses,
             human_stats.draws);
  }
};
SessionStats session_stats;

}
using namespace chess;

// Forward declarations for functions defined later
void update_session_stats(GameMode mode, bool white_won, bool black_won, bool is_draw);
std::string uimove_to_uci(const UIMove &move);

// Global variables
#if PUFFER_REPLAY_ENABLED
extern bool show_game_list;
#else
bool show_game_list = false;
#endif

// Function Implementations
#if PUFFER_REPLAY_ENABLED
void load_available_games() {
  available_games.clear();
  if (global_game_logger) {
    global_game_logger->load_games_from_directory();
    const auto &games = global_game_logger->get_games();
    for (size_t i = 0; i < games.size(); ++i) {
      char buffer[256];
      std::string outcome_display = games[i].outcome;
      snprintf(buffer, sizeof(buffer), "Game %zu: %s (%d moves) - %s", i + 1,
               outcome_display.c_str(), games[i].total_moves,
               games[i].timestamp.c_str());
      available_games.push_back(std::string(buffer));
    }
  }
}
void render_game_list_screen() {
  ClearBackground(RL_RAYWHITE);
  DrawText("Game Replay - Select Game", 50, 20, 24, RL_BLACK);
  if (available_games.empty()) {
    DrawText("No games found in logs.", 50, 100, 18, RL_RED);
  } else {
    for (size_t i = 0; i < available_games.size(); ++i) {
      ::Color color = (i == (size_t)selected_game_index) ? RL_RED : RL_BLACK;
      DrawText(available_games[i].c_str(), 50, 80 + i * 20, 14, color);
    }
  }
}
void render_game_replay_screen(CChess *env, ChessPieceTextures *textures) {
  ClearBackground(RL_RAYWHITE);
  if (!current_replay.is_active)
    return;
  c_reset(env);
  for (int i = 0; i <= current_replay.current_move_index; ++i) {
    const auto &game_move = current_replay.current_game->moves[i];
    char uci_can[6];
    if (env->ctx->board.to_move == C_BLACK)
      flip_uci_for_black_perspective(ACTION_ID_TO_UCI[game_move.action_id],
                                     uci_can);
    else
      strcpy(uci_can, ACTION_ID_TO_UCI[game_move.action_id]);
    apply_uci_move(env->ctx, uci_can);
  }
  render_chess_board(env, textures);
  //... Replay UI
}
void handle_game_list_input() {
  if (IsKeyPressed(KEY_UP))
    selected_game_index = std::max(0, selected_game_index - 1);
  if (IsKeyPressed(KEY_DOWN))
    selected_game_index =
        std::min((int)available_games.size() - 1, selected_game_index + 1);
  if (IsKeyPressed(KEY_ENTER) && !available_games.empty()) {
    current_replay.start_replay(
        &global_game_logger->get_games()[selected_game_index]);
    replay_mode_active = true;
    show_game_list = false;
  }
  if (IsKeyPressed(KEY_B)) {
    show_game_list = false;
  }
}
void handle_game_replay_input() {
  if (!current_replay.is_active)
    return;
  if (IsKeyPressed(KEY_RIGHT))
    current_replay.next_move();
  if (IsKeyPressed(KEY_LEFT))
    current_replay.prev_move();
  if (IsKeyPressed(KEY_B)) {
    replay_mode_active = false;
    show_game_list = true;
  }
}
#else
// Stubs for when replay is disabled
void load_available_games() {}
void render_game_list_screen() {
  ClearBackground(RL_RAYWHITE);
  DrawText("Game Replay Disabled", 50, 20, 24, RL_DARKGRAY);
  DrawText("json.hpp not found during compilation.", 50, 60, 16, RL_DARKGRAY);
}
void render_game_replay_screen(CChess *env, ChessPieceTextures *textures) {}
void handle_game_list_input() {
  if (IsKeyPressed(KEY_B)) {
    show_game_list = false;
  }
}
void handle_game_replay_input() {}
bool replay_mode_active = false; // Must be defined
#endif

void check_and_update_game_outcome(CChess *env, GameMode mode) {
  if (env->terminals[0]) {
    bool white_won = env->log.white_win > 0.5;
    bool black_won = env->log.black_win > 0.5;
    bool is_draw = env->log.game_drawn > 0.5;
    update_session_stats(mode, white_won, black_won, is_draw);
    game_moves.clear();
  }
}

void update_session_stats(GameMode mode, bool white_won, bool black_won,
                          bool is_draw) {
  session_stats.total_games++;
  if (white_won) {
    session_stats.total_wins++;
    session_stats.white_stats.add_win();
    session_stats.black_stats.add_loss();
  } else if (black_won) {
    session_stats.total_losses++;
    session_stats.white_stats.add_loss();
    session_stats.black_stats.add_win();
  } else if (is_draw) {
    session_stats.total_draws++;
    session_stats.white_stats.add_draw();
    session_stats.black_stats.add_draw();
  }

  if (mode == GM_PLAYER_STOCKFISH || mode == GM_PLAYER_RANDOM) {
    if (white_won)
      session_stats.human_stats.add_win();
    else if (black_won)
      session_stats.human_stats.add_loss();
    else
      session_stats.human_stats.add_draw();
  } else if (mode == GM_AGENT_STOCKFISH || mode == GM_AGENT_AGENT ||
             mode == GM_AGENT_RANDOM) {
    if (white_won)
      session_stats.agent_stats.add_win();
    else if (black_won)
      session_stats.agent_stats.add_loss();
    else
      session_stats.agent_stats.add_draw();
  } else if (mode == GM_RANDOM_AGENT) {
    if (black_won)
      session_stats.agent_stats.add_win();
    else if (white_won)
      session_stats.agent_stats.add_loss();
    else
      session_stats.agent_stats.add_draw();
  }
}

int agent_select_action(ChessNet *net, CChess *env, int agent_idx) {
  if (!net || !env)
    return 0;
  int action;
  const int obs_size = 1344 + TOTAL_CHESS_ACTIONS; // Board encoding + action mask
  forward_chessnet(net, env->observations + (agent_idx * obs_size), &action);
  return action;
}

int random_select_action(CChess *env) {
  chess_generate_legal_moves_uci(env->ctx);
  printf("[DEBUG] random_select_action: to_move=%s, legal_moves_count=%d\n", 
         env->ctx->board.to_move == C_WHITE ? "WHITE" : "BLACK", 
         env->ctx->legal_moves_count);
  
  if (env->ctx->legal_moves_count == 0)
    return 0;
    
  std::uniform_int_distribution<int> dist(0, env->ctx->legal_moves_count - 1);
  std::random_device rd;
  int move_idx = dist(rd);
  const char *uci_move = env->ctx->legal_moves_buffer[move_idx];
  
  printf("[DEBUG] random_select_action: selected move_idx=%d, uci_move='%s'\n", 
         move_idx, uci_move);
  
  // Must match action mask creation logic: flip canonical move to perspective for BLACK
  char perspective_uci[6];
  int action_id;
  if (env->ctx->board.to_move == C_BLACK) {
    flip_uci_for_black_perspective(uci_move, perspective_uci);
    action_id = uci_to_action_id(perspective_uci);
    printf("[DEBUG] random_select_action: BLACK canonical='%s' -> perspective='%s' -> action_id=%d\n", 
           uci_move, perspective_uci, action_id);
  } else {
    action_id = uci_to_action_id(uci_move);
    printf("[DEBUG] random_select_action: WHITE canonical='%s' -> action_id=%d\n", 
           uci_move, action_id);
  }
  
  return action_id;
}

// Check if a move is a promotion by using the environment's move generation system
bool is_promotion_move(CChess *env, int from_x, int from_y, int to_x, int to_y) {
  printf("[DEBUG] is_promotion_move: checking (%d,%d) to (%d,%d)\n", from_x, from_y, to_x, to_y);
  
  // Check if from coordinates are valid
  if (from_x < 0 || from_x >= 8 || from_y < 0 || from_y >= 8) {
    printf("[DEBUG] is_promotion_move: invalid from coordinates\n");
    return false;
  }
  
  // Check if to coordinates are valid  
  if (to_x < 0 || to_x >= 8 || to_y < 0 || to_y >= 8) {
    printf("[DEBUG] is_promotion_move: invalid to coordinates\n");
    return false;
  }
  
  // Get the piece at the from square using the environment's board state
  Piece from_piece = env->ctx->board.board[from_y * 8 + from_x];
  printf("[DEBUG] is_promotion_move: piece at (%d,%d) = type:%d, color:%d\n", from_x, from_y, from_piece.type, from_piece.color);
  
  // Check if it's a pawn move
  if (from_piece.type != PAWN) {
    printf("[DEBUG] is_promotion_move: not a pawn move\n");
    return false;
  }
  
  // Check if it's moving to the promotion rank
  if (from_piece.color == C_WHITE && to_y == 7) {
    printf("[DEBUG] is_promotion_move: WHITE pawn reaching rank 8 - PROMOTION!\n");
    return true;
  }
  if (from_piece.color == C_BLACK && to_y == 0) {
    printf("[DEBUG] is_promotion_move: BLACK pawn reaching rank 1 - PROMOTION!\n");
    return true;
  }
  
  printf("[DEBUG] is_promotion_move: pawn not reaching promotion rank\n");
  return false;
}

void render_promotion_selection() {
  if (!show_promotion_selection) return;
  
  // Draw semi-transparent overlay
  DrawRectangle(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT, ColorAlpha(RL_BLACK, 0.6f));
  
  // Draw promotion selection dialog
  int dialog_width = 320;
  int dialog_height = 160;
  int dialog_x = (WINDOW_WIDTH - dialog_width) / 2;
  int dialog_y = (WINDOW_HEIGHT - dialog_height) / 2;
  
  DrawRectangle(dialog_x, dialog_y, dialog_width, dialog_height, LIGHTGRAY);
  DrawRectangleLines(dialog_x, dialog_y, dialog_width, dialog_height, DARKGRAY);
  
  // Draw title
  DrawText("Choose Promotion Piece", dialog_x + 20, dialog_y + 20, 20, DARKGRAY);
  
  // Draw promotion piece options with text labels
  const char* promotion_names[] = {"Queen", "Rook", "Bishop", "Knight"};
  PieceType promotion_types[] = {QUEEN, ROOK, BISHOP, KNIGHT};
  
  for (int i = 0; i < 4; i++) {
    int piece_x = dialog_x + 20 + i * 70;
    int piece_y = dialog_y + 60;
    
    // Highlight selected piece
    Color highlight_color = (selected_promotion == promotion_types[i]) ? BLUE : GRAY;
    DrawRectangle(piece_x - 2, piece_y - 2, 64 + 4, 64 + 4, highlight_color);
    DrawRectangle(piece_x, piece_y, 64, 64, RL_WHITE);
    
    // Draw piece name
    DrawText(promotion_names[i], piece_x + 5, piece_y + 10, 12, RL_BLACK);
    
    // Draw hotkey
    DrawText(TextFormat("%d", i + 1), piece_x + 50, piece_y + 30, 14, RL_BLACK);
  }
  
  // Draw instructions
  DrawText("Press 1=Queen, 2=Rook, 3=Bishop, 4=Knight", dialog_x + 20, dialog_y + dialog_height - 30, 12, DARKGRAY);
}

void handle_promotion_selection() {
  if (!show_promotion_selection) {
    return;
  }
  
  printf("[DEBUG] handle_promotion_selection: dialog active, waiting for input\n");
  printf("[DEBUG] handle_promotion_selection: Press 1=Queen, 2=Rook, 3=Bishop, 4=Knight\n");
  
  // Debug: Check if any keys are being pressed
  for (int key = 32; key < 127; key++) {
    if (IsKeyPressed(key)) {
      printf("[DEBUG] handle_promotion_selection: Key pressed: %d (char: '%c')\n", key, (char)key);
    }
  }
  
  // Check special keys
  if (IsKeyPressed(KEY_ENTER)) printf("[DEBUG] handle_promotion_selection: KEY_ENTER detected\n");
  if (IsKeyPressed(KEY_SPACE)) printf("[DEBUG] handle_promotion_selection: KEY_SPACE detected\n");
  if (IsKeyPressed(KEY_KP_ENTER)) printf("[DEBUG] handle_promotion_selection: KEY_KP_ENTER detected\n");
  
  if (IsKeyPressed(KEY_ONE)) {
    printf("[DEBUG] handle_promotion_selection: QUEEN selected - executing immediately\n");
    selected_promotion = QUEEN;
    show_promotion_selection = false;
    UIMove promoted_move = {
        {(int8_t)promotion_from_x, (int8_t)promotion_from_y},
        {(int8_t)promotion_to_x, (int8_t)promotion_to_y},
        selected_promotion};
    std::string uci = uimove_to_uci(promoted_move);
    printf("[DEBUG] handle_promotion_selection: promotion UCI='%s'\n", uci.c_str());
    global_env_ptr->actions[0] = uci_to_action_id(uci.c_str());
    game_moves.push_back(uci);
    c_step(global_env_ptr);
    promotion_from_x = -1;
    promotion_from_y = -1;
    promotion_to_x = -1;
    promotion_to_y = -1;
    selected_promotion = QUEEN;
    printf("[DEBUG] handle_promotion_selection: QUEEN promotion completed\n");
  }
  if (IsKeyPressed(KEY_TWO)) {
    printf("[DEBUG] handle_promotion_selection: ROOK selected - executing immediately\n");
    selected_promotion = ROOK;
    show_promotion_selection = false;
    UIMove promoted_move = {
        {(int8_t)promotion_from_x, (int8_t)promotion_from_y},
        {(int8_t)promotion_to_x, (int8_t)promotion_to_y},
        selected_promotion};
    std::string uci = uimove_to_uci(promoted_move);
    printf("[DEBUG] handle_promotion_selection: promotion UCI='%s'\n", uci.c_str());
    global_env_ptr->actions[0] = uci_to_action_id(uci.c_str());
    game_moves.push_back(uci);
    c_step(global_env_ptr);
    promotion_from_x = -1;
    promotion_from_y = -1;
    promotion_to_x = -1;
    promotion_to_y = -1;
    selected_promotion = QUEEN;
    printf("[DEBUG] handle_promotion_selection: ROOK promotion completed\n");
  }
  if (IsKeyPressed(KEY_THREE)) {
    printf("[DEBUG] handle_promotion_selection: BISHOP selected - executing immediately\n");
    selected_promotion = BISHOP;
    show_promotion_selection = false;
    UIMove promoted_move = {
        {(int8_t)promotion_from_x, (int8_t)promotion_from_y},
        {(int8_t)promotion_to_x, (int8_t)promotion_to_y},
        selected_promotion};
    std::string uci = uimove_to_uci(promoted_move);
    printf("[DEBUG] handle_promotion_selection: promotion UCI='%s'\n", uci.c_str());
    global_env_ptr->actions[0] = uci_to_action_id(uci.c_str());
    game_moves.push_back(uci);
    c_step(global_env_ptr);
    promotion_from_x = -1;
    promotion_from_y = -1;
    promotion_to_x = -1;
    promotion_to_y = -1;
    selected_promotion = QUEEN;
    printf("[DEBUG] handle_promotion_selection: BISHOP promotion completed\n");
  }
  if (IsKeyPressed(KEY_FOUR)) {
    printf("[DEBUG] handle_promotion_selection: KNIGHT selected - executing immediately\n");
    selected_promotion = KNIGHT;
    show_promotion_selection = false;
    UIMove promoted_move = {
        {(int8_t)promotion_from_x, (int8_t)promotion_from_y},
        {(int8_t)promotion_to_x, (int8_t)promotion_to_y},
        selected_promotion};
    std::string uci = uimove_to_uci(promoted_move);
    printf("[DEBUG] handle_promotion_selection: promotion UCI='%s'\n", uci.c_str());
    global_env_ptr->actions[0] = uci_to_action_id(uci.c_str());
    game_moves.push_back(uci);
    c_step(global_env_ptr);
    promotion_from_x = -1;
    promotion_from_y = -1;
    promotion_to_x = -1;
    promotion_to_y = -1;
    selected_promotion = QUEEN;
    printf("[DEBUG] handle_promotion_selection: KNIGHT promotion completed\n");
  }
}

std::string uimove_to_uci(const UIMove &move) {
  char uci_str[6];
  char promo_char = ' ';
  
  // Debug output
  printf("[DEBUG] uimove_to_uci: from=(%d,%d) to=(%d,%d)\n", 
         move.from.x, move.from.y, move.to.x, move.to.y);
  
  if (move.promotion != EMPTY) {
    switch (move.promotion) {
    case QUEEN:
      promo_char = 'q';
      break;
    case ROOK:
      promo_char = 'r';
      break;
    case BISHOP:
      promo_char = 'b';
      break;
    case KNIGHT:
      promo_char = 'n';
      break;
    default:
      break;
    }
    snprintf(uci_str, 6, "%c%c%c%c%c", 'a' + move.from.x, '1' + move.from.y,
             'a' + move.to.x, '1' + move.to.y, promo_char);
  } else {
    snprintf(uci_str, 5, "%c%c%c%c", 'a' + move.from.x, '1' + move.from.y,
             'a' + move.to.x, '1' + move.to.y);
  }
  
  printf("[DEBUG] uimove_to_uci: generated UCI = '%s'\n", uci_str);
  return std::string(uci_str);
}

void render_chess_board(CChess *env, ChessPieceTextures *textures) {
  if (!env || !textures || !env->ctx)
    return;
  for (int y = 0; y < 8; ++y) {
    for (int x = 0; x < 8; ++x) {
      int screen_x = BOARD_OFFSET_X + x * SQUARE_SIZE;
      int screen_y = BOARD_OFFSET_Y + (7 - y) * SQUARE_SIZE;
      ::Color square_color = ((x + y) % 2 == 0) ? RL_BEIGE : RL_BROWN;
      DrawRectangle(screen_x, screen_y, SQUARE_SIZE, SQUARE_SIZE, square_color);
      const Piece *piece = get_piece_const(&env->ctx->board, x, y);
      if (piece && piece->type != EMPTY) {
        Texture2D piece_texture =
            get_piece_texture(textures, piece->color, piece->type);
        if (piece_texture.id > 0) {
          DrawTexturePro(
              piece_texture,
              {0, 0, (float)piece_texture.width, (float)piece_texture.height},
              {(float)screen_x, (float)screen_y, (float)SQUARE_SIZE,
               (float)SQUARE_SIZE},
              {0, 0}, 0.0f, RL_WHITE);
        }
      }
    }
  }
  DrawRectangleLines(BOARD_OFFSET_X - 2, BOARD_OFFSET_Y - 2, BOARD_SIZE + 4,
                     BOARD_SIZE + 4, RL_BLACK);
}

// int main() {
//   printf("PufferLib Chess Evaluation – GUI Menu Version\n");
//   srand(time(NULL));

// #if PUFFER_REPLAY_ENABLED
//   global_game_logger =
//       new GameLogger("pufferlib/resources/chess/training_logs/complete_games");
// #endif
// 
//   const char *weights_path = "resources/chess/puffer_chess_weights.bin";
//   Weights *weights = NULL;
//   
//   // Try to load weights, but continue with zero weights if file doesn't exist
//   FILE *weight_file = fopen(weights_path, "rb");
//   if (weight_file) {
//     fclose(weight_file);
//     weights = load_weights(weights_path, CHESS_NUM_WEIGHTS);
//     if (!weights) {
//       fprintf(stderr, "ERROR: Could not load weights from %s\n", weights_path);
//       return 1;
//     }
//     printf("Loaded pre-trained weights from %s\n", weights_path);
//   } else {
//     printf("No pre-trained weights found at %s, initializing with zero weights\n", weights_path);
//     weights = (Weights*)calloc(1, sizeof(Weights) + CHESS_NUM_WEIGHTS*sizeof(float));
//     weights->data = (float*)(weights + 1);
//     weights->size = CHESS_NUM_WEIGHTS;
//     weights->idx = 0;
//     // Initialize with small random values for better training
//     for (int i = 0; i < CHESS_NUM_WEIGHTS; i++) {
//       weights->data[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.02f;
//     }
//   }
//   ChessNet *agent_net = init_chessnet(weights, 2);
// 
//   InitWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "PufferLib Chess – Menu");
//   SetTargetFPS(60);
//   ChessPieceTextures textures = load_piece_textures();
// 
//   CChess env = {0};
//   global_env_ptr = &env;
//   int elo_setting = 1320;
//   bool in_menu = true;
//   int menu_index = 0;
//   GameMode game_mode = GM_PLAYER_STOCKFISH;
// 
//   while (!WindowShouldClose()) {
//     BeginDrawing();
//     if (in_menu) {
//       ClearBackground(RL_RAYWHITE);
//       DrawText("PufferLib Chess", 50, 20, 32, RL_BLACK);
//       for (int i = 0; i < GM_COUNT; ++i) {
//         ::Color col = (i == menu_index) ? RL_RED : RL_BLACK;
// #if !PUFFER_REPLAY_ENABLED
//         if (i == GM_GAME_REPLAY)
//           col = RL_DARKGRAY;
// #endif
//         DrawText(GAME_MODE_NAMES[i], 80, 120 + i * 30, 20, col);
//       }
//       if (IsKeyPressed(KEY_UP))
//         menu_index = (menu_index + GM_COUNT - 1) % GM_COUNT;
//       if (IsKeyPressed(KEY_DOWN))
//         menu_index = (menu_index + 1) % GM_COUNT;
//       if (IsKeyPressed(KEY_ENTER)) {
//         game_mode = static_cast<GameMode>(menu_index);
// #if !PUFFER_REPLAY_ENABLED
//         if (game_mode == GM_GAME_REPLAY)
//           continue;
// #endif
//         game_moves.clear();
//         in_menu = false;
// 
//         env.max_depth = 500;
//         allocate(&env);
//         set_dual_agent_self_play_mode(&env, game_mode == GM_AGENT_AGENT);
//         c_reset(&env);
// 
//         if (game_mode == GM_GAME_REPLAY) {
//           load_available_games();
//           show_game_list = true;
//         }
//       }
//     } else if (show_game_list) {
//       render_game_list_screen();
//       handle_game_list_input();
//       if (!show_game_list && !replay_mode_active)
//         in_menu = true;
//     } else if (replay_mode_active) {
//       render_game_replay_screen(&env, &textures);
//       handle_game_replay_input();
//       if (!replay_mode_active)
//         show_game_list = true;
//     } else {
//       // Gameplay loop
//       if (IsKeyPressed(KEY_M)) {
//         in_menu = true;
//         free_allocated(&env);
//         continue;
//       }
//       if (IsKeyPressed(KEY_R)) {
//         game_moves.clear();
//         c_reset(&env);
//       }
// 
//       auto *ctx = env.ctx;
//       bool is_human_turn =
//           (game_mode == GM_PLAYER_STOCKFISH || game_mode == GM_PLAYER_RANDOM) &&
//           ctx->board.to_move == C_WHITE;
// 
//       if (!game_paused && !env.terminals[0] && !is_human_turn) {
//         compute_observation_with_perspective(&env, ctx);
//         int action = 0;
//         int agent_idx = (ctx->board.to_move == C_WHITE) ? 0 : 1;
// 
//         if (game_mode == GM_AGENT_AGENT ||
//             (game_mode == GM_AGENT_STOCKFISH &&
//              ctx->board.to_move == C_WHITE) ||
//             (game_mode == GM_RANDOM_AGENT && ctx->board.to_move == C_BLACK)) {
//           action = agent_select_action(agent_net, &env, agent_idx);
//         } else if (game_mode == GM_PLAYER_STOCKFISH) {
//           // Stockfish moves handled by c_step
//         } else {
//           action = random_select_action(&env);
//         }
// 
//         env.actions[agent_idx] = action;
//         c_step(&env);
//         check_and_update_game_outcome(&env, game_mode);
//       }
// 
//       if (is_human_turn && !env.terminals[0]) {
//         // Handle promotion selection if dialog is active
//         if (show_promotion_selection) {
//           handle_promotion_selection();
//         } else if (IsMouseButtonPressed(MOUSE_LEFT_BUTTON)) {
//           static int sel_fx = -1, sel_fy = -1;
//           Vector2 mp = GetMousePosition();
//           int bx = (mp.x - BOARD_OFFSET_X) / SQUARE_SIZE;
//           int screen_rank = ((mp.y - BOARD_OFFSET_Y) / SQUARE_SIZE);
//           int by = 7 - screen_rank;
//           printf("[DEBUG] Mouse pos: (%.1f, %.1f), bx=%d, screen_rank=%d, by=%d\n", 
//                  mp.x, mp.y, bx, screen_rank, by);
//           if (bx >= 0 && bx < 8 && by >= 0 && by < 8) {
//             if (sel_fx == -1) {
//               const Piece *p = get_piece_const(&ctx->board, bx, by);
//               if (p && p->type != EMPTY && p->color == C_WHITE) {
//                 sel_fx = bx;
//                 sel_fy = by;
//               }
//             } else {
//               printf("[DEBUG] Mouse click: checking move from (%d,%d) to (%d,%d)\n", sel_fx, sel_fy, bx, by);
//               if (is_promotion_move(&env, sel_fx, sel_fy, bx, by)) {
//                 printf("[DEBUG] Mouse click: PROMOTION MOVE DETECTED - setting show_promotion_selection=true\n");
//                 show_promotion_selection = true;
//                 promotion_from_x = sel_fx;
//                 promotion_from_y = sel_fy;
//                 promotion_to_x = bx;
//                 promotion_to_y = by;
//               } else {
//                 printf("[DEBUG] Mouse click: REGULAR MOVE - processing normally\n");
//                 UIMove move = {{(int8_t)sel_fx, (int8_t)sel_fy},
//                                {(int8_t)bx, (int8_t)by},
//                                EMPTY};
//                 std::string uci = uimove_to_uci(move);
//                 int action_id = uci_to_action_id(uci.c_str());
//                 printf("[DEBUG] Mouse click: sel_fx=%d, sel_fy=%d, bx=%d, by=%d\n", sel_fx, sel_fy, bx, by);
//                 printf("[DEBUG] UCI='%s' -> action_id=%d\n", uci.c_str(), action_id);
//                 env.actions[0] = action_id;
//                 c_step(&env);
//                 check_and_update_game_outcome(&env, game_mode);
//               }
//               sel_fx = -1;
//               sel_fy = -1;
//             }
//           }
//         }
//       }
// 
//       ClearBackground(RL_RAYWHITE);
//       render_chess_board(&env, &textures);
//       
//       // Render promotion selection dialog if active
//       if (show_promotion_selection) {
//         render_promotion_selection();
//       }
//     }
//     EndDrawing();
//   }
// 
// #if PUFFER_REPLAY_ENABLED
//   delete global_game_logger;
// #endif
//   
//   // Close Raylib window first, which will automatically cleanup textures
// //   CloseWindow();
// //   
// //   // Skip manual memory cleanup - let the OS handle it on process exit
// //   // Manual cleanup can cause double-free issues with shared references
// //   return 0;
// // }

// #include <stdio.h>
// #include <stdlib.h>
// #include <string.h>
// #include <time.h>

// // The header for the chess environment.
// #include "chess.h"
// #include "raylib.h"

// Forward declarations for local functions
void test_performance(float test_time);
int demo();

/**
 * @brief Performance test for the CChess environment.
 * @param test_time The duration of the performance test in seconds.
 */
void test_performance(float test_time) {
  const int num_envs = 2048;

  CChess *envs = (CChess *)malloc(sizeof(CChess) * num_envs);
  if (!envs) {
    printf("Failed to allocate memory for environments.\n");
    return;
  }

  printf("Allocating and initializing %d chess environments...\n", num_envs);

  // Initialize each environment in the batch
  for (int i = 0; i < num_envs; i++) {
    // --- CRITICAL FIX: SET CONFIGURATION *BEFORE* ALLOCATE ---
    // The allocate() function calls init(), which copies these values.
    // They MUST be set first.
    envs[i].max_depth = 500;
    envs[i].reward_valid = 0.0f;
    envs[i].reward_invalid_white = 0.0f;
    envs[i].reward_invalid_black = 0.0f;
    envs[i].reward_agent_captures_enemy_piece = 0.0f;
    envs[i].reward_enemy_captures_agent_piece = 0.0f;
    envs[i].reward_draw = 0.0f;
    envs[i].reward_win_white = 1.0f;
    envs[i].reward_loss_white = -1.0f;
    envs[i].reward_win_black = 1.0f;
    envs[i].reward_loss_black = -1.0f;
    envs[i].reward_check_white = 0.0f;
    envs[i].reward_check_black = 0.0f;
    envs[i].reward_material_diff_white = 0.0f;
    envs[i].reward_material_diff_black = 0.0f;
    envs[i].debug_disable_mask = false;
    envs[i].stockfish_enabled = false;

    // Now that configuration is set, it's safe to allocate.
    allocate(&envs[i]);

    // Reset the environment to the starting game state.
    c_reset(&envs[i]);
  }

  printf("Starting realistic performance test for %.1f seconds...\n",
         test_time);

  // Allocate a buffer to store legal actions for one environment.
  // This is done once to avoid repeated allocation in the hot loop.
  int *legal_actions = (int *)malloc(sizeof(int) * TOTAL_CHESS_ACTIONS);
  if (!legal_actions) {
    printf("Failed to allocate memory for legal_actions buffer.\n");
    for (int i = 0; i < num_envs; i++) {
      free_allocated(&envs[i]);
    }
    free(envs);
    return;
  }

  time_t start_time = time(NULL);
  long long total_batch_steps = 0;

  // Main performance loop
  while (time(NULL) - start_time < test_time) {
    for (int i = 0; i < num_envs; i++) {
      // --- INTEGRATED REALISTIC ACTION SELECTION ---
      int num_legal_actions = 0;

      // The observation contains the action mask at the end.
      // The offset is the size of the board planes (21 planes * 8x8 squares = 1344).
      const int MASK_OFFSET = 1344;
      float *mask = &envs[i].observations[MASK_OFFSET];

      // Build a list of all legal action indices from the mask.
      for (int j = 0; j < TOTAL_CHESS_ACTIONS; j++) {
        if (mask[j] == 1.0f) {
          legal_actions[num_legal_actions++] = j;
        }
      }

      // If there are legal moves, pick one at random.
      // Otherwise, the game is over and will be reset by c_step,
      // so the next action doesn't matter. We can default to 0.
      if (num_legal_actions > 0) {
        envs[i].actions[0] = legal_actions[rand() % num_legal_actions];
      } else {
        envs[i].actions[0] = 0; // Fallback for terminal states
      }

      c_step(&envs[i]);
    }
    total_batch_steps++;
  }
  time_t end_time = time(NULL);

  // Free the legal actions buffer
  free(legal_actions);

  float elapsed_time = (float)(end_time - start_time);
  if (elapsed_time < 1)
    elapsed_time = 1;

  double total_individual_steps = (double)num_envs * total_batch_steps;
  double sps = total_individual_steps / elapsed_time;

  printf("\n--- Performance Test Results ---\n");
  printf("Elapsed time: %.2f seconds\n", elapsed_time);
  printf("Total individual steps: %.0f\n", total_individual_steps);
  printf("Steps Per Second (SPS): %.2f\n", sps);
  printf("--------------------------------\n");

  for (int i = 0; i < num_envs; i++) {
    free_allocated(&envs[i]);
  }
  free(envs);
}

// /**
//  * @brief A simple graphical demo of the chess environment using Raylib.
//  *
//  * @return int Returns 0 on successful execution.
//  */
// int demo() {
//   CChess env = {};
//   env.max_depth = 500;
//   allocate(&env);
//   c_reset(&env);

//   const int screenWidth = 400;
//   const int screenHeight = 450;
//   InitWindow(screenWidth, screenHeight, "Chess Demo (Random Moves)");
//   SetTargetFPS(5);

//   while (!WindowShouldClose()) {
//     // Only process a move if the game is not over
//     if (env.terminals[0] == 0) {
//       // Generate legal moves to populate the cache
//       chess_generate_legal_moves_uci(env.ctx);

//       if (env.ctx->legal_moves_count > 0) {
//         int move_idx = rand() % env.ctx->legal_moves_count;
//         // The policy network would receive a flipped board and choose a
//         "white"
//         // move. The environment expects the action from the current player's
//         // perspective. Since we are not using a policy, we directly pick the
//         // correct action ID.
//         if (env.ctx->board.to_move == C_WHITE) {
//           env.actions[0] =
//               uci_to_action_id(env.ctx->legal_moves_buffer[move_idx]);
//         } else {
//           char perspective_uci[6];
//           flip_uci_for_black_perspective(env.ctx->legal_moves_buffer[move_idx],
//                                          perspective_uci);
//           env.actions[0] = uci_to_action_id(perspective_uci);
//         }
//       } else {
//         env.actions[0] = 0; // No legal moves available
//       }
//       c_step(&env);
//     } else {
//       // If the game is over, reset it after a short delay for viewing
//       // In a real scenario, this would be handled by the training loop
//       // For the demo, we just reset it to keep it running
//       c_reset(&env);
//     }

//     BeginDrawing();
//     ClearBackground(RAYWHITE);

//     for (int y = 0; y < 8; y++) {
//       for (int x = 0; x < 8; x++) {
//         Color squareColor = ((x + y) % 2 == 0) ? BEIGE : BROWN;
//         DrawRectangle(x * 50, y * 50, 50, 50, squareColor);

//         const Piece *p = get_piece_const(&env.ctx->board, x, 7 - y);
//         if (p && p->type != EMPTY) {
//           const char *piece_text;
//           switch (p->type) {
//           case KING:
//             piece_text = (p->color == C_WHITE) ? "K" : "k";
//             break;
//           case QUEEN:
//             piece_text = (p->color == C_WHITE) ? "Q" : "q";
//             break;
//           case ROOK:
//             piece_text = (p->color == C_WHITE) ? "R" : "r";
//             break;
//           case BISHOP:
//             piece_text = (p->color == C_WHITE) ? "B" : "b";
//             break;
//           case KNIGHT:
//             piece_text = (p->color == C_WHITE) ? "N" : "n";
//             break;
//           case PAWN:
//             piece_text = (p->color == C_WHITE) ? "P" : "p";
//             break;
//           default:
//             piece_text = "?";
//             break;
//           }
//           Color piece_color = (p->color == C_WHITE) ? WHITE : BLACK;
//           DrawText(piece_text, x * 50 + 18, y * 50 + 10, 30, piece_color);
//         }
//       }
//     }

//     const char *to_move_text =
//         (env.ctx->board.to_move == C_WHITE) ? "White to move" : "Black to
//         move";
//     DrawText(to_move_text, 10, 410, 20, BLACK);

//     if (env.terminals[0] != 0) {
//       DrawText("GAME OVER", 120, 410, 20, RED);
//     }

//     EndDrawing();
//   }

//   CloseWindow();
//   free_allocated(&env);
//   return 0;
// }

int main(int argc, char **argv) {
  srand(time(NULL));

  if (argc > 1 && strcmp(argv[1], "demo") == 0) {
    printf("Demo mode not available in this build\n");
    // demo();
  } else {
    printf("Running performance test...\n");
    test_performance(30);
  }

  return 0;
}