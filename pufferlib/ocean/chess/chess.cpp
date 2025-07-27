// chess.cpp - Graphical Chess Evaluation using Raylib
#include <algorithm>
#include <cctype>
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

#ifdef __cplusplus
extern "C" {
#endif
// PufferLib C headers for the neural network
#include "../../extensions/puffernet.h"
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
static const Color RL_DARKRED = MAROON;
static const Color RL_PURPLE = PURPLE;

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

#define CHESS_NUM_WEIGHTS 2016433

static inline void mask_logits(float *logits, const float *legal, int size) {
  for (int i = 0; i < size; ++i) {
    if (legal[i] < 0.5f)
      logits[i] = -1e9f;
  }
}

static ChessNet *init_chessnet(Weights *weights, int num_agents) {
  ChessNet *net = (ChessNet *)calloc(1, sizeof(ChessNet));
  net->num_agents = num_agents;
  net->board_enc1 = make_linear(weights, num_agents, 1472, 512);
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

static void reset_lstm_state(ChessNet *net) {
  // Reset LSTM hidden and cell states to zero at game start
  int state_size = 256; // From chess.ini: hidden_size = 256
  memset(net->lstm->state_h, 0, state_size * sizeof(float));
  memset(net->lstm->state_c, 0, state_size * sizeof(float));
  printf("[DEBUG] LSTM state reset to zero\n");
}

static void forward_chessnet(ChessNet *net, float *observations, int *actions) {
  float *board_obs = observations;
  
  // Convert sparse action mask to dense format
  // observations[1472] = num_legal_moves
  // observations[1473:1537] = action_ids (up to 64)
  static float dense_legal_mask[TOTAL_CHESS_ACTIONS];
  memset(dense_legal_mask, 0, sizeof(dense_legal_mask));
  
  int num_legal_moves = (int)observations[1472];
  if (num_legal_moves > 0 && num_legal_moves <= 64) {
    for (int i = 0; i < num_legal_moves; i++) {
      int action_id = (int)observations[1473 + i];
      if (action_id >= 0 && action_id < TOTAL_CHESS_ACTIONS) {
        dense_legal_mask[action_id] = 1.0f;
      }
    }
  }
  
  float *legal_mask = dense_legal_mask;
  
  linear(net->board_enc1, board_obs);
  relu(net->board_relu1, net->board_enc1->output);
  linear(net->board_enc2, net->board_relu1->output);
  relu(net->board_relu2, net->board_enc2->output);
  linear(net->combiner, net->board_relu2->output);
  relu(net->comb_relu, net->combiner->output);
  lstm(net->lstm, net->comb_relu->output);
  linear(net->policy_head, net->lstm->state_h);
  
  // Debug: check if network is producing meaningful outputs BEFORE masking
  float min_logit = net->policy_head->output[0];
  float max_logit = net->policy_head->output[0];
  for (int i = 1; i < TOTAL_CHESS_ACTIONS; i++) {
    if (net->policy_head->output[i] < min_logit) min_logit = net->policy_head->output[i];
    if (net->policy_head->output[i] > max_logit) max_logit = net->policy_head->output[i];
  }
  
  int legal_count = 0;
  for (int i = 0; i < TOTAL_CHESS_ACTIONS; i++) {
    if (legal_mask[i] > 0.5f) legal_count++;
  }
  
  printf("[DEBUG] forward_chessnet: PRE-MASK logits range=[%.6f, %.6f], legal_moves=%d\n",
         min_logit, max_logit, legal_count);
  
  // Use the 1968-dimensional legal mask directly
  mask_logits(net->policy_head->output, legal_mask, TOTAL_CHESS_ACTIONS);
  
  softmax_multidiscrete(net->md, net->policy_head->output, actions);
  
  printf("[DEBUG] forward_chessnet: softmax_multidiscrete returned actions[0]=%d (max allowed: %d)\n", 
         actions[0], TOTAL_CHESS_ACTIONS - 1);
  
  // Ensure output action is within valid range
  if (actions[0] >= TOTAL_CHESS_ACTIONS || actions[0] < 0) {
    printf("[WARNING] Neural network selected invalid action %d (valid range: 0-%d), clamping to 0\n", 
           actions[0], TOTAL_CHESS_ACTIONS - 1);
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
  
  // Skip texture loading if graphics context is not ready
  if (!IsWindowReady()) {
    printf("Warning: Window not ready, skipping texture loading\n");
    return textures;
  }
  
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
  
  // Verify at least one texture loaded successfully
  bool any_loaded = (textures.wking.id > 0 || textures.wqueen.id > 0 || 
                     textures.wrook.id > 0 || textures.wpawn.id > 0);
  
  if (!any_loaded) {
    printf("Warning: No chess piece textures loaded successfully\n");
  } else {
    printf("Chess piece textures loaded successfully\n");
  }
  
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
int current_move_index = -1;  // -1 means at current game position, 0+ means viewing history
bool viewing_history = false; // true when using arrow keys to navigate
bool game_ending_processed = false; // prevent multiple game ending processing
bool show_promotion_selection = false;
int promotion_from_x = -1, promotion_from_y = -1, promotion_to_x = -1,
    promotion_to_y = -1;
PieceType selected_promotion = QUEEN;
CChess *global_env_ptr = nullptr;
ChessNet *global_agent_net = nullptr;

// Game mode definitions
enum GameMode {
  GM_PLAYER_AGENT,
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
const char *GAME_MODE_NAMES[] = {"Player vs Agent",     "Player vs Stockfish", "Player vs Random",
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
void apply_moves_to_current_position(CChess *env, int up_to_move);

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
    // Calculate scrolling offset to keep selected game visible
    const int max_visible_games = 25; // Max games that fit on screen
    int scroll_offset = 0;
    if ((int)available_games.size() > max_visible_games) {
      scroll_offset = std::max(0, selected_game_index - max_visible_games / 2);
      scroll_offset = std::min(scroll_offset, (int)available_games.size() - max_visible_games);
    }
    
    int end_index = std::min((int)available_games.size(), scroll_offset + max_visible_games);
    for (int i = scroll_offset; i < end_index; ++i) {
      ::Color color = (i == selected_game_index) ? RL_RED : RL_BLACK;
      DrawText(available_games[i].c_str(), 50, 80 + (i - scroll_offset) * 20, 14, color);
    }
    
    // Show scroll indicator if there are more games
    if ((int)available_games.size() > max_visible_games) {
      char scroll_info[64];
      snprintf(scroll_info, sizeof(scroll_info), "Showing %d-%d of %d games (UP/DOWN to scroll)", 
               scroll_offset + 1, end_index, (int)available_games.size());
      DrawText(scroll_info, 50, 60, 12, RL_DARKGRAY);
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
    const char* uci_move_white_perspective = ACTION_ID_TO_UCI[game_move.action_id];
    
    // Convert from white perspective to canonical move for apply_uci_move
    char canonical_uci[6];
    const char* player_name;
    if ((i % 2) == 1) { // Black's move (odd indices are black moves)
      // For black moves, the stored action_id represents white perspective,
      // so we flip it to get the canonical move
      flip_uci_for_black_perspective(uci_move_white_perspective, canonical_uci);
      player_name = "BLACK";
    } else { // White's move (even indices are white moves)
      // For white moves, white perspective IS canonical
      strcpy(canonical_uci, uci_move_white_perspective);
      player_name = "WHITE";
    }
    
    printf("[REPLAY_DEBUG] Move %d (%s): action_id=%d -> white_perspective='%s' -> canonical='%s'\n", 
           i+1, player_name, game_move.action_id, uci_move_white_perspective, canonical_uci);
    
    // Print board state before move
    printf("[REPLAY_DEBUG] Board before move %d:\n", i+1);
    for (int rank = 7; rank >= 0; rank--) {
      printf("[REPLAY_DEBUG] ");
      for (int file = 0; file < 8; file++) {
        Piece* piece = get_piece(&env->ctx->board, file, rank);
        char piece_char = '.';
        if (piece && piece->type != EMPTY) {
          char piece_symbols[] = ".KQRBNP";
          piece_char = piece_symbols[piece->type];
          if (piece->color == C_BLACK) piece_char = tolower(piece_char);
        }
        printf("%c", piece_char);
      }
      printf(" %d\n", rank + 1);
    }
    printf("[REPLAY_DEBUG] abcdefgh\n");
    
    apply_uci_move(env->ctx, canonical_uci);
    
    // Print board state after move
    printf("[REPLAY_DEBUG] Board after applying '%s':\n", canonical_uci);
    for (int rank = 7; rank >= 0; rank--) {
      printf("[REPLAY_DEBUG] ");
      for (int file = 0; file < 8; file++) {
        Piece* piece = get_piece(&env->ctx->board, file, rank);
        char piece_char = '.';
        if (piece && piece->type != EMPTY) {
          char piece_symbols[] = ".KQRBNP";
          piece_char = piece_symbols[piece->type];
          if (piece->color == C_BLACK) piece_char = tolower(piece_char);
        }
        printf("%c", piece_char);
      }
      printf(" %d\n", rank + 1);
    }
    printf("[REPLAY_DEBUG] abcdefgh\n");
    printf("[REPLAY_DEBUG] -----------\n");
  }
  render_chess_board(env, textures);
  //... Replay UI
}
void handle_game_list_input() {
  // Use IsKeyDown for rapid navigation when holding arrow keys
  if (IsKeyDown(KEY_UP))
    selected_game_index = std::max(0, selected_game_index - 1);
  if (IsKeyDown(KEY_DOWN))
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
  if (IsKeyPressed(KEY_R)) {
    // Reset to main game by exiting game list and clearing flags
    show_game_list = false;
    replay_mode_active = false;
    // Also reset the game state when available
    if (global_env_ptr) {
      game_moves.clear();
      viewing_history = false;
      current_move_index = -1;
      game_ending_processed = false;
      c_reset(global_env_ptr);
      if (global_agent_net) {
        reset_lstm_state(global_agent_net);
      }
    }
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
  if (IsKeyPressed(KEY_R)) {
    // Reset to main game by exiting replay mode and clearing flags
    replay_mode_active = false;
    show_game_list = false;
    // Also reset the game state when available
    if (global_env_ptr) {
      game_moves.clear();
      viewing_history = false;
      current_move_index = -1;
      game_ending_processed = false;
      c_reset(global_env_ptr);
      if (global_agent_net) {
        reset_lstm_state(global_agent_net);
      }
    }
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
  if (IsKeyPressed(KEY_R)) {
    // Reset to main game by exiting game list
    show_game_list = false;
    // Also reset the game state when available
    if (global_env_ptr) {
      game_moves.clear();
      viewing_history = false;
      current_move_index = -1;
      game_ending_processed = false;
      c_reset(global_env_ptr);
      if (global_agent_net) {
        reset_lstm_state(global_agent_net);
      }
    }
  }
}
void handle_game_replay_input() {}
bool replay_mode_active = false; // Must be defined
#endif

void check_and_update_game_outcome(CChess *env, GameMode mode) {
  printf("[DEBUG] check_and_update_game_outcome called: terminals[0]=%d, game_ending_processed=%d\n", 
         env->terminals[0], game_ending_processed);
  printf("[DEBUG] Global flag: game_just_ended=%d, white_won=%d, black_won=%d, is_draw=%d\n",
         last_game_outcome.game_ended, last_game_outcome.white_won, last_game_outcome.black_won, last_game_outcome.is_draw);
  
  // Check if game just ended by looking at terminals[0] directly
  if (env->terminals[0] && !game_ending_processed) {
    // Game just ended - determine the outcome from the log counters
    bool white_won = (env->log.white_win > 0);
    bool black_won = (env->log.black_win > 0);
    bool is_draw = (env->log.game_drawn > 0);
    
    printf("[DEBUG] Direct terminal detection: white_win=%d, black_win=%d, draw=%d\n",
           env->log.white_win, env->log.black_win, env->log.game_drawn);
           
    // Set the outcome for processing
    last_game_outcome.game_ended = true;
    last_game_outcome.white_won = white_won;
    last_game_outcome.black_won = black_won;
    last_game_outcome.is_draw = is_draw;
  }
  
  if (last_game_outcome.game_ended && !game_ending_processed) {
    bool white_won = last_game_outcome.white_won;
    bool black_won = last_game_outcome.black_won;
    bool is_draw = last_game_outcome.is_draw;
    
    // Clear the flag so we don't process this game end multiple times
    last_game_outcome.game_ended = false;
    
    // Auto-pause the game when it ends (but only for human vs modes, not agent vs agent/random)
    if (mode == GM_PLAYER_AGENT || mode == GM_PLAYER_STOCKFISH || mode == GM_PLAYER_RANDOM) {
      game_paused = true;
    }
    
    // Log the final position (terminal state reached) - only once
    printf("[GAME END] Final position after %d moves:\n", (int)game_moves.size());
    const char* result_str = white_won ? "WHITE WINS" : (black_won ? "BLACK WINS" : "DRAW");
    printf("[GAME END] Result: %s\n", result_str);
    
    // Print the final move sequence for reference
    for (size_t i = 0; i < game_moves.size(); i++) {
      if (i % 2 == 0) {
        printf("%d. %s ", (int)(i/2) + 1, game_moves[i].c_str());
      } else {
        printf("%s\n", game_moves[i].c_str());
      }
    }
    if (game_moves.size() % 2 == 1) printf("\n");
    
    update_session_stats(mode, white_won, black_won, is_draw);
    
    // Mark as processed to prevent multiple calls for this game
    game_ending_processed = true;
    
    // For automatic game modes (agent vs agent/random), reset the flag after a short delay
    // so the next game can be processed. For human modes, keep it set until manual reset.
    if (mode == GM_AGENT_AGENT || mode == GM_AGENT_RANDOM || mode == GM_AGENT_STOCKFISH || mode == GM_RANDOM_AGENT) {
      // Reset the flag so the next game can be detected
      // We do this immediately since auto-reset will start a new game right away
      game_ending_processed = false;
    }
    
    // DON'T clear game_moves - keep them for navigation
    // game_moves will be cleared only when starting a new game (R key or menu)
  }
}

void update_session_stats(GameMode mode, bool white_won, bool black_won,
                          bool is_draw) {
  printf("[STATS DEBUG] update_session_stats called: mode=%d, white_won=%d, black_won=%d, is_draw=%d\n", 
         mode, white_won, black_won, is_draw);
  printf("[STATS DEBUG] Before update: total_games=%d, agent_games=%d, human_games=%d\n",
         session_stats.total_games, session_stats.agent_stats.games, session_stats.human_stats.games);
  
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

  if (mode == GM_PLAYER_AGENT || mode == GM_PLAYER_STOCKFISH || mode == GM_PLAYER_RANDOM) {
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
  
  printf("[STATS DEBUG] After update: total_games=%d, agent_games=%d, human_games=%d\n",
         session_stats.total_games, session_stats.agent_stats.games, session_stats.human_stats.games);
  printf("[STATS DEBUG] Agent stats: wins=%d, losses=%d, draws=%d\n",
         session_stats.agent_stats.wins, session_stats.agent_stats.losses, session_stats.agent_stats.draws);
  printf("[STATS DEBUG] Human stats: wins=%d, losses=%d, draws=%d\n",
         session_stats.human_stats.wins, session_stats.human_stats.losses, session_stats.human_stats.draws);
}

int agent_select_action(ChessNet *net, CChess *env, int agent_idx) {
  if (!net || !env) {
    return 0;
  }
  
  int action;
  
  // Quick observation sanity check
  float board_sum = 0.0f;
  for (int i = 0; i < 1472; i++) board_sum += env->observations[i];
  int num_legal_moves = (int)env->observations[1472];
  printf("[AGENT] Board pieces: %.0f, Legal moves: %d\n", board_sum, num_legal_moves);
  
  // If no legal moves, the game should be over - don't call the neural network
  if (num_legal_moves == 0) {
    printf("[AGENT] No legal moves available - game should be terminal\n");
    return 0; // Return any action, it won't be used since game is over
  }
  
  // Standard Ocean pattern: trust env->observations (computed by environment)
  forward_chessnet(net, env->observations, &action);
  
  return action;
}

int random_select_action(CChess *env) {
  // Use sparse action mask: observations[1472] = count, observations[1473:1537] = action IDs
  int num_legal_moves = (int)env->observations[1472];
  
  if (num_legal_moves == 0) {
    printf("[RANDOM] No legal moves available - game should be terminal\n");
    return 0;
  }
  
  // Select random legal action from the sparse list
  int random_index = rand() % num_legal_moves;
  int action = (int)env->observations[1473 + random_index];
  
  printf("[RANDOM] Selected action %d from %d legal moves\n", action, num_legal_moves);
  return action;
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

void apply_moves_to_current_position(CChess *env, int up_to_move) {
  // Reset to starting position
  c_reset(env);
  
  // Apply moves one by one to reach the desired position
  for (int i = 0; i < up_to_move && i < (int)game_moves.size(); i++) {
    const char* uci_move = game_moves[i].c_str();
    int action_id = uci_to_action_id(uci_move);
    
    if (action_id >= 0 && action_id < TOTAL_CHESS_ACTIONS) {
      env->actions[0] = action_id;
      c_step(env);
    }
  }
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
//   GameMode game_mode = GM_PLAYER_AGENT;
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
//           action = agent_select_action(agent_net, &env, 0);  // Standard Ocean pattern
//         } else if (game_mode == GM_PLAYER_STOCKFISH) {
//           // Stockfish moves handled by c_step
//         } else {
//           action = random_select_action(&env);
//         }
// 
//         env.actions[0] = action;  // FIXED: c_step always reads from actions[0]
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
//           if (bx >= 0 && bx < 8 && by >= 0 && by < 8) {
//             if (sel_fx == -1) {
//               const Piece *p = get_piece_const(&ctx->board, bx, by);
//               if (p && p->type != EMPTY && p->color == C_WHITE) {
//                 sel_fx = bx;
//                 sel_fy = by;
//               }
//             } else {
//               if (is_promotion_move(&env, sel_fx, sel_fy, bx, by)) {
//                 show_promotion_selection = true;
//                 promotion_from_x = sel_fx;
//                 promotion_from_y = sel_fy;
//                 promotion_to_x = bx;
//                 promotion_to_y = by;
//               } else {
//                 UIMove move = {{(int8_t)sel_fx, (int8_t)sel_fy},
//                                {(int8_t)bx, (int8_t)by},
//                                EMPTY};
//                 std::string uci = uimove_to_uci(move);
//                 int action_id = uci_to_action_id(uci.c_str());
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
    // envs[i].reward_agent_captures_enemy_piece = 0.0f;
    // envs[i].reward_enemy_captures_agent_piece = 0.0f;
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

// Main demo function with full UI
int demo() {
  printf("PufferLib Chess Evaluation – GUI Menu Version\n");
  srand(time(NULL));

#if PUFFER_REPLAY_ENABLED
  global_game_logger =
      new GameLogger("pufferlib/resources/chess/training_logs/complete_games");
#endif
 
  const char *weights_path = "pufferlib/resources/chess/puffer_chess_weights.bin";
  Weights *weights = NULL;
  ChessNet *agent_net = NULL;
  
  // Try to load weights, but continue with zero weights if file doesn't exist
  FILE *weight_file = fopen(weights_path, "rb");
  if (weight_file) {
    fclose(weight_file);
    weights = load_weights(weights_path, CHESS_NUM_WEIGHTS);
    if (!weights) {
      fprintf(stderr, "ERROR: Could not load weights from %s\n", weights_path);
      return 1;
    }
    printf("Loaded pre-trained weights from %s\n", weights_path);
  } else {
    printf("No pre-trained weights found at %s, initializing with zero weights\n", weights_path);
    weights = (Weights*)calloc(1, sizeof(Weights) + CHESS_NUM_WEIGHTS*sizeof(float));
    weights->data = (float*)(weights + 1);
    weights->size = CHESS_NUM_WEIGHTS;
    weights->idx = 0;
    // Initialize with small random values for better training
    for (int i = 0; i < CHESS_NUM_WEIGHTS; i++) {
      weights->data[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.02f;
    }
  }
  agent_net = init_chessnet(weights, 2);
  global_agent_net = agent_net;
 
  // Initialize graphics with safety checks
  SetConfigFlags(FLAG_WINDOW_RESIZABLE | FLAG_MSAA_4X_HINT);
  InitWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "PufferLib Chess – Menu");
  
  if (!IsWindowReady()) {
    printf("ERROR: Failed to initialize graphics window\n");
    return 1;
  }
  
  SetTargetFPS(60);
  
  // Load textures with error checking
  ChessPieceTextures textures = {0};
  printf("Loading chess piece textures...\n");
  textures = load_piece_textures();
  printf("Texture loading completed.\n");
 
  CChess env = {0};
  global_env_ptr = &env;
  int elo_setting = 1320;
  bool in_menu = true;
  int menu_index = 0;
  GameMode game_mode = GM_PLAYER_AGENT;
  bool minimal_rendering = false; // Fallback for graphics issues
 
  while (!WindowShouldClose()) {
    BeginDrawing();
    if (in_menu) {
      ClearBackground(RL_RAYWHITE);
      DrawText("PufferLib Chess", 50, 20, 32, RL_BLACK);
      DrawText("Controls: UP/DOWN arrows to navigate, ENTER to select", 50, 60, 16, RL_DARKGRAY);
      DrawText("In game: M=Menu, R=Reset, Mouse=Move pieces", 50, 80, 16, RL_DARKGRAY);
      
      for (int i = 0; i < GM_COUNT; ++i) {
        ::Color col = (i == menu_index) ? RL_RED : RL_BLACK;
#if !PUFFER_REPLAY_ENABLED
        if (i == GM_GAME_REPLAY)
          col = RL_DARKGRAY;
#endif
        DrawText(GAME_MODE_NAMES[i], 80, 120 + i * 30, 20, col);
      }
      
      // Debug: show current menu index
      char debug_text[64];
      snprintf(debug_text, sizeof(debug_text), "Current selection: %d", menu_index);
      DrawText(debug_text, 80, 400, 16, RL_BLUE);
      // Use IsKeyPressed for single key presses, or IsKeyDown with timing for held keys
      static float key_repeat_timer = 0.0f;
      static bool key_held = false;
      
      if (IsKeyPressed(KEY_UP)) {
        menu_index = (menu_index + GM_COUNT - 1) % GM_COUNT;
        key_repeat_timer = 0.0f;
        key_held = true;
      } else if (IsKeyPressed(KEY_DOWN)) {
        menu_index = (menu_index + 1) % GM_COUNT;
        key_repeat_timer = 0.0f;
        key_held = true;
      } else if (IsKeyDown(KEY_UP) && key_held) {
        key_repeat_timer += GetFrameTime();
        if (key_repeat_timer > 0.3f) { // 300ms delay before repeating
          menu_index = (menu_index + GM_COUNT - 1) % GM_COUNT;
          key_repeat_timer = 0.2f; // Reset to 200ms for subsequent repeats
        }
      } else if (IsKeyDown(KEY_DOWN) && key_held) {
        key_repeat_timer += GetFrameTime();
        if (key_repeat_timer > 0.3f) { // 300ms delay before repeating
          menu_index = (menu_index + 1) % GM_COUNT;
          key_repeat_timer = 0.2f; // Reset to 200ms for subsequent repeats
        }
      } else if (!IsKeyDown(KEY_UP) && !IsKeyDown(KEY_DOWN)) {
        key_held = false;
        key_repeat_timer = 0.0f;
      }
      if (IsKeyPressed(KEY_ENTER)) {
        game_mode = static_cast<GameMode>(menu_index);
        printf("[DEBUG] Selected menu index: %d, game_mode: %d (%s)\n", menu_index, game_mode, GAME_MODE_NAMES[menu_index]);
#if !PUFFER_REPLAY_ENABLED
        if (game_mode == GM_GAME_REPLAY)
          continue;
#endif
        game_moves.clear();
        viewing_history = false;
        current_move_index = -1;
        game_paused = false; // Reset pause state
        game_ending_processed = false; // Reset game ending flag
        
        // Reset game outcome flags for new game
        last_game_outcome.game_ended = false;
        last_game_outcome.white_won = false;
        last_game_outcome.black_won = false;
        last_game_outcome.is_draw = false;
        
        in_menu = false;
 
        env.max_depth = 500;
        allocate(&env);
        set_dual_agent_self_play_mode(&env, game_mode == GM_AGENT_AGENT);
        c_reset(&env);
        
        // Reset LSTM state at start of new game
        if (agent_net) {
          reset_lstm_state(agent_net);
        }
 
        if (game_mode == GM_GAME_REPLAY) {
          load_available_games();
          show_game_list = true;
        }
      }
    } else if (show_game_list) {
      render_game_list_screen();
      handle_game_list_input();
      if (!show_game_list && !replay_mode_active)
        in_menu = true;
    } else if (replay_mode_active) {
      render_game_replay_screen(&env, &textures);
      handle_game_replay_input();
      if (!replay_mode_active)
        show_game_list = true;
    } else {
      // Gameplay loop
      if (IsKeyPressed(KEY_M)) {
        in_menu = true;
        free_allocated(&env);
        continue;
      }
      if (IsKeyPressed(KEY_R)) {
        game_moves.clear();
        viewing_history = false;
        current_move_index = -1;
        game_paused = false; // Reset pause state
        game_ending_processed = false; // Reset game ending flag
        
        // Reset game outcome flags for new game
        last_game_outcome.game_ended = false;
        last_game_outcome.white_won = false;
        last_game_outcome.black_won = false;
        last_game_outcome.is_draw = false;
        
        // Properly reinitialize the environment (same as starting from menu)
        free_allocated(&env);
        env.max_depth = 500;
        allocate(&env);
        set_dual_agent_self_play_mode(&env, game_mode == GM_AGENT_AGENT);
        c_reset(&env);
        
        // Reset LSTM state when restarting game
        if (agent_net) {
          reset_lstm_state(agent_net);
        }
      }
      if (IsKeyPressed(KEY_P)) {
        game_paused = !game_paused;
      }
      
      // Arrow key navigation through move history (like browser feature)
      if (IsKeyDown(KEY_LEFT) && !game_moves.empty()) {
        if (!viewing_history) {
          // Start viewing history from the last move
          current_move_index = game_moves.size() - 1;
          viewing_history = true;
        } else if (current_move_index > 0) {
          current_move_index--;
        }
        // Apply moves up to current_move_index to show that position
        apply_moves_to_current_position(&env, current_move_index + 1);
      }
      if (IsKeyDown(KEY_RIGHT) && viewing_history) {
        if (current_move_index < (int)game_moves.size() - 1) {
          current_move_index++;
          // Apply moves up to current_move_index to show that position  
          apply_moves_to_current_position(&env, current_move_index + 1);
        } else {
          // Return to live game position
          viewing_history = false;
          current_move_index = -1;
          // Restore the current game state by applying all moves
          apply_moves_to_current_position(&env, game_moves.size());
        }
      }
      
      // Add HOME and END keys for quick navigation like browser
      if (IsKeyPressed(KEY_HOME) && !game_moves.empty()) {
        current_move_index = 0;
        viewing_history = true;
        apply_moves_to_current_position(&env, 0); // Show starting position
      }
      if (IsKeyPressed(KEY_END) && !game_moves.empty()) {
        viewing_history = false;
        current_move_index = -1;
        apply_moves_to_current_position(&env, game_moves.size()); // Show final position
      }
 
      bool is_human_turn = false;
      if (env.ctx != nullptr) {
        is_human_turn =
            (game_mode == GM_PLAYER_AGENT || game_mode == GM_PLAYER_STOCKFISH || game_mode == GM_PLAYER_RANDOM) &&
            env.ctx->board.to_move == C_WHITE;
      }
 
      if (!game_paused && !env.terminals[0] && !is_human_turn && env.ctx != nullptr) {
        // Ensure observations are computed before agent acts (needed for network inference)
        compute_observation_with_perspective(&env, env.ctx);
        
        int action = 0;
        int agent_idx = (env.ctx->board.to_move == C_WHITE) ? 0 : 1;
 
        if (game_mode == GM_AGENT_AGENT ||
            (game_mode == GM_AGENT_STOCKFISH && env.ctx->board.to_move == C_WHITE) ||
            (game_mode == GM_AGENT_RANDOM && env.ctx->board.to_move == C_WHITE) ||
            (game_mode == GM_RANDOM_AGENT && env.ctx->board.to_move == C_BLACK) ||
            (game_mode == GM_PLAYER_AGENT && env.ctx->board.to_move == C_BLACK)) {
          action = agent_select_action(agent_net, &env, 0);  // Standard Ocean pattern
          
          printf("[AGENT] Selected action %d\n", action);
        } else if (game_mode == GM_PLAYER_STOCKFISH) {
          // Stockfish moves handled by c_step
        } else {
          action = random_select_action(&env);
        }
 
        // Check if we should skip c_step due to game ending conditions
        int num_legal_moves = (int)env.observations[1472];
        
        // Check for various terminal conditions
        bool should_force_terminal = false;
        bool white_won = false, black_won = false, is_draw = false;
        
        if (num_legal_moves == 0) {
          // No legal moves - checkmate or stalemate
          should_force_terminal = true;
          if (env.ctx->board.to_move == C_WHITE) {
            black_won = true;
            printf("[GAME] Black wins - White has no legal moves\n");
          } else {
            white_won = true;
            printf("[GAME] White wins - Black has no legal moves\n");
          }
        }
        // Add 50-move rule check (halfmove_clock >= 100)
        else if (env.ctx->halfmove_clock >= 100) {
          should_force_terminal = true;
          is_draw = true;
          printf("[GAME] Draw by 50-move rule (halfmove_clock=%d)\n", env.ctx->halfmove_clock);
        }
        // Add step limit check (max_depth reached)
        else if (env.step_count >= env.max_depth) {
          should_force_terminal = true;
          is_draw = true;
          printf("[GAME] Draw by step limit (step_count=%d, max_depth=%d)\n", env.step_count, env.max_depth);
        }
        
        if (should_force_terminal) {
          printf("[GAME] Forcing game termination\n");
          // Manually set terminal state and outcome flags
          env.terminals[0] = 1;
          
          // Set appropriate log counters based on outcome
          if (white_won) {
            env.log.white_win = 1;
          } else if (black_won) {
            env.log.black_win = 1;
          } else if (is_draw) {
            env.log.game_drawn = 1;
          }
          
          // Immediately set the global outcome flag for stats processing
          last_game_outcome.game_ended = true;
          last_game_outcome.white_won = white_won;
          last_game_outcome.black_won = black_won;
          last_game_outcome.is_draw = is_draw;
        } else {
          // Standard Ocean pattern: simple action assignment + c_step
          env.actions[0] = action;
          printf("[DEBUG] About to call c_step with env=%p\n", (void*)&env);
          
          // Debug: Print board state before move (only for castling moves)
          const char* uci_move = (action >= 0 && action < TOTAL_CHESS_ACTIONS) ? ACTION_ID_TO_UCI[action] : "invalid";
          if (uci_move && (strstr(uci_move, "e1g1") || strstr(uci_move, "e1c1") || strstr(uci_move, "e8g8") || strstr(uci_move, "e8c8"))) {
            printf("[CASTLING] Before %s: observing position...\n", uci_move);
            compute_observation_with_perspective(&env, env.ctx);
          }
          
          c_step(&env);
          
          // Debug: Print board state after move (only for castling moves)
          if (uci_move && (strstr(uci_move, "e1g1") || strstr(uci_move, "e1c1") || strstr(uci_move, "e8g8") || strstr(uci_move, "e8c8"))) {
            printf("[CASTLING] After %s: re-observing position...\n", uci_move);
            compute_observation_with_perspective(&env, env.ctx);
          }
        }
        
        // Check for game termination immediately after c_step (before auto-reset)
        // The core logic will set terminals[0]=1 then auto-reset, so we need to detect it here
        check_and_update_game_outcome(&env, game_mode);
        
        // Add back GUI functionality: move recording and stats (after c_step)
        // Use ACTION_ID_TO_UCI mapping for reliable move recording
        if (action >= 0 && action < TOTAL_CHESS_ACTIONS) {
          const char* uci_move = ACTION_ID_TO_UCI[action];
          if (uci_move && strlen(uci_move) > 0) {
            printf("[DEBUG] Action %d -> UCI: %s\n", action, uci_move);
            game_moves.push_back(std::string(uci_move));
            printf("[MOVE] Played: %s (action %d)\n", uci_move, action);
          } else {
            printf("[DEBUG] Action %d has no valid UCI mapping\n", action);
          }
        } else {
          printf("[DEBUG] Action %d is out of range [0, %d)\n", action, TOTAL_CHESS_ACTIONS);
        }
        
        check_and_update_game_outcome(&env, game_mode);
        
        // Add a small delay in fully automated modes to ensure keyboard input can be processed
        if (game_mode == GM_AGENT_AGENT || game_mode == GM_AGENT_RANDOM || 
            game_mode == GM_RANDOM_RANDOM || game_mode == GM_RANDOM_AGENT) {
          // Wait a few frames to allow input processing
          static int frame_delay_counter = 0;
          frame_delay_counter++;
          if (frame_delay_counter < 3) {
            // Skip to next frame iteration to allow input processing
            goto render_frame;
          }
          frame_delay_counter = 0;
        }
      }
 
      if (is_human_turn && !env.terminals[0]) {
        // Handle promotion selection if dialog is active
        if (show_promotion_selection) {
          handle_promotion_selection();
        } else if (IsMouseButtonPressed(MOUSE_LEFT_BUTTON)) {
          static int sel_fx = -1, sel_fy = -1;
          Vector2 mp = GetMousePosition();
          int bx = (mp.x - BOARD_OFFSET_X) / SQUARE_SIZE;
          int screen_rank = ((mp.y - BOARD_OFFSET_Y) / SQUARE_SIZE);
          int by = 7 - screen_rank;
          printf("[DEBUG] Mouse pos: (%.1f, %.1f), bx=%d, screen_rank=%d, by=%d\n", 
                 mp.x, mp.y, bx, screen_rank, by);
          if (bx >= 0 && bx < 8 && by >= 0 && by < 8 && env.ctx != nullptr) {
            if (sel_fx == -1) {
              const Piece *p = get_piece_const(&env.ctx->board, bx, by);
              if (p && p->type != EMPTY && p->color == C_WHITE) {
                sel_fx = bx;
                sel_fy = by;
              }
            } else {
              printf("[DEBUG] Mouse click: checking move from (%d,%d) to (%d,%d)\n", sel_fx, sel_fy, bx, by);
              if (is_promotion_move(&env, sel_fx, sel_fy, bx, by)) {
                printf("[DEBUG] Mouse click: PROMOTION MOVE DETECTED - setting show_promotion_selection=true\n");
                show_promotion_selection = true;
                promotion_from_x = sel_fx;
                promotion_from_y = sel_fy;
                promotion_to_x = bx;
                promotion_to_y = by;
              } else {
                printf("[DEBUG] Mouse click: REGULAR MOVE - processing normally\n");
                UIMove move = {{(int8_t)sel_fx, (int8_t)sel_fy},
                               {(int8_t)bx, (int8_t)by},
                               EMPTY};
                std::string uci = uimove_to_uci(move);
                int action_id = uci_to_action_id(uci.c_str());
                printf("[DEBUG] Mouse click: sel_fx=%d, sel_fy=%d, bx=%d, by=%d\n", sel_fx, sel_fy, bx, by);
                printf("[DEBUG] UCI='%s' -> action_id=%d\n", uci.c_str(), action_id);
                env.actions[0] = action_id;
                c_step(&env);
                
                // Record human move in game history
                game_moves.push_back(uci);
                printf("[MOVE] Human played: %s (action %d)\n", uci.c_str(), action_id);
                
                check_and_update_game_outcome(&env, game_mode);
              }
              sel_fx = -1;
              sel_fy = -1;
            }
          }
        }
      }
 
      render_frame:
      ClearBackground(RL_RAYWHITE);
      render_chess_board(&env, &textures);
      
      // Add controls display below chess board (1/2 square width spacing)
      int controls_y = BOARD_OFFSET_Y + BOARD_SIZE + (SQUARE_SIZE / 2);
      DrawText("Controls:", BOARD_OFFSET_X, controls_y, 16, RL_BLACK);
      DrawText("M = Menu, R = Reset, P = Pause", BOARD_OFFSET_X, controls_y + 20, 14, RL_DARKGRAY);
      DrawText("Mouse = Select/Move pieces", BOARD_OFFSET_X, controls_y + 40, 14, RL_DARKGRAY);
      
      // Display game type above the board
      char mode_text[64];
      snprintf(mode_text, sizeof(mode_text), "%s", GAME_MODE_NAMES[game_mode]);
      DrawText(mode_text, BOARD_OFFSET_X, 20, 20, RL_BLACK);
      
      // Display current turn and move number above the board
      if (env.ctx != nullptr) {
        const char* turn_text = env.ctx->board.to_move == C_WHITE ? "White's Turn" : "Black's Turn";
        char move_num_text[64];
        if (viewing_history) {
          snprintf(move_num_text, sizeof(move_num_text), "Viewing: Move %d/%d (arrows to navigate)", 
                  current_move_index + 1, (int)game_moves.size());
        } else {
          snprintf(move_num_text, sizeof(move_num_text), "Move #%d", (env.ctx->board.fullmove_number));
        }
        
        if (!env.terminals[0] && !viewing_history) {
          DrawText(turn_text, BOARD_OFFSET_X, 45, 16, env.ctx->board.to_move == C_WHITE ? RL_BLUE : RL_DARKBLUE);
        }
        
        ::Color move_color = viewing_history ? RL_ORANGE : RL_DARKGRAY;
        DrawText(move_num_text, BOARD_OFFSET_X + 150, 45, 14, move_color);
      }
      
      // Add instruction text for navigation
      if (env.terminals[0] && !game_moves.empty()) {
        DrawText("LEFT/RIGHT: Review moves (hold for rapid), HOME/END: Start/Final position", BOARD_OFFSET_X, 650, 14, RL_DARKGRAY);
      }
      
      // Display move history (moved right 2 squares and down 1 square from 450,220)
      if (!game_moves.empty()) {
        int recent_moves_x = 450 + (2 * SQUARE_SIZE);  // 578
        int recent_moves_y = 220 + SQUARE_SIZE;        // 284
        DrawText("Recent Moves:", recent_moves_x, recent_moves_y, 16, RL_BLACK);
        int moves_to_show = game_moves.size() < 16 ? game_moves.size() : 16; // Show up to 8 move pairs
        int display_line = 0;
        for (int i = game_moves.size() - moves_to_show; i < (int)game_moves.size() && display_line < 8; i += 2) {
          char move_text[64];
          int move_number = (i / 2) + 1;
          if (i + 1 < (int)game_moves.size()) {
            // Both white and black moves available
            snprintf(move_text, sizeof(move_text), "%d. %s %s", move_number, 
                    game_moves[i].c_str(), game_moves[i + 1].c_str());
          } else {
            // Only white move available
            snprintf(move_text, sizeof(move_text), "%d. %s", move_number, game_moves[i].c_str());
          }
          DrawText(move_text, recent_moves_x, recent_moves_y + 20 + display_line * 18, 13, RL_DARKGRAY);
          display_line++;
        }
      }
      
      // Show game result if game is over - NO AUTO RESET (moved 2 squares right)
      if (env.terminals[0]) {
        int game_over_x = 450 + (2 * SQUARE_SIZE);  // 578
        DrawText("GAME OVER:", game_over_x, 380, 18, RL_BLACK);
        if (last_game_outcome.white_won) {
          DrawText("WHITE WINS!", game_over_x, 405, 20, RL_DARKGREEN);
        } else if (last_game_outcome.black_won) {
          DrawText("BLACK WINS!", game_over_x, 405, 20, RL_DARKGREEN);
        } else if (last_game_outcome.is_draw) {
          char draw_text[128];
          const char* draw_reason = last_game_outcome.draw_reason.empty() ? "Unknown" : last_game_outcome.draw_reason.c_str();
          snprintf(draw_text, sizeof(draw_text), "DRAW: %s", draw_reason);
          DrawText(draw_text, game_over_x, 405, 16, RL_ORANGE);
        }
        DrawText("Press R to play again or M for menu", game_over_x, 430, 14, RL_DARKGRAY);
        
        // Show game statistics
        char move_count_text[32];
        snprintf(move_count_text, sizeof(move_count_text), "Total moves: %d", (int)game_moves.size());
        DrawText(move_count_text, 580, 450, 14, RL_DARKGRAY);
      }
      
      // Display session statistics (moved to right of recent moves to avoid overlap)
      if (session_stats.total_games >= 0 && session_stats.total_games < 10000) {
        int session_stats_x = 550 + SQUARE_SIZE; // Move right by one chess board square width
        DrawText("Session Stats:", session_stats_x, 20, 16, RL_BLACK);
        char games_text[32], wins_text[32], losses_text[32], draws_text[32];
        snprintf(games_text, sizeof(games_text), "Games: %d", session_stats.total_games);
        snprintf(wins_text, sizeof(wins_text), "Wins: %d", session_stats.total_wins);
        snprintf(losses_text, sizeof(losses_text), "Losses: %d", session_stats.total_losses);
        snprintf(draws_text, sizeof(draws_text), "Draws: %d", session_stats.total_draws);
        DrawText(games_text, session_stats_x, 40, 14, RL_DARKGRAY);
        DrawText(wins_text, session_stats_x, 60, 14, RL_DARKGREEN);
        DrawText(losses_text, session_stats_x, 80, 14, RL_DARKRED);
        DrawText(draws_text, session_stats_x, 100, 14, RL_ORANGE);
        
        // Debug: Show game state
        char debug_text[128];
        snprintf(debug_text, sizeof(debug_text), "DEBUG: term=%d, end_proc=%d, mode=%d, steps=%d/%d", 
                env.terminals[0], game_ending_processed, game_mode,
                env.ctx ? env.ctx->step_count : -1, env.max_depth);
        DrawText(debug_text, session_stats_x, 120, 10, RL_DARKGRAY);
        
        // Debug: Print session stats to console every few frames
        static int debug_counter = 0;
        if (debug_counter++ % 300 == 0) { // Every ~5 seconds at 60fps
          printf("[STATS DEBUG] Session: games=%d, wins=%d, losses=%d, draws=%d\n",
                 session_stats.total_games, session_stats.total_wins, 
                 session_stats.total_losses, session_stats.total_draws);
        }
        
        // Show both player and agent stats separately (if they have games played)
        if (session_stats.human_stats.games > 0) {
          DrawText("Player Stats:", session_stats_x, 130, 14, RL_BLACK);
          char human_stats_text[64];
          snprintf(human_stats_text, sizeof(human_stats_text), "%.1f%% (%d-%d-%d)", 
                   session_stats.human_stats.win_rate() * 100,
                   session_stats.human_stats.wins,
                   session_stats.human_stats.losses,
                   session_stats.human_stats.draws);
          DrawText(human_stats_text, session_stats_x, 150, 12, RL_BLUE);
        }
        
        if (session_stats.agent_stats.games > 0) {
          // Position agent stats below player stats if both exist
          int agent_y_offset = (session_stats.human_stats.games > 0) ? 40 : 0;
          DrawText("Agent Stats:", session_stats_x, 130 + agent_y_offset, 14, RL_BLACK);
          char agent_stats_text[64];
          snprintf(agent_stats_text, sizeof(agent_stats_text), "%.1f%% (%d-%d-%d)", 
                   session_stats.agent_stats.win_rate() * 100,
                   session_stats.agent_stats.wins,
                   session_stats.agent_stats.losses,
                   session_stats.agent_stats.draws);
          DrawText(agent_stats_text, session_stats_x, 150 + agent_y_offset, 12, RL_PURPLE);
        }
      }
      
      // Render promotion selection dialog if active
      if (show_promotion_selection) {
        render_promotion_selection();
      }
    }
    EndDrawing();
  }
 
#if PUFFER_REPLAY_ENABLED
  delete global_game_logger;
#endif
  
  // Properly clean up neural network - do NOT free textures manually
  if (agent_net) {
    free_chessnet(agent_net);
    agent_net = NULL;
    global_agent_net = NULL;
  }
  if (weights) {
    free(weights);
    weights = NULL;
  }
  
  // Let Raylib handle texture cleanup automatically in CloseWindow()
  CloseWindow();
  
  // Only free if not already freed (to prevent double free)
  if (!in_menu) {
    free_allocated(&env);
  }
  return 0;
}

// Console-only chess demo for testing without graphics
int demo_console() {
  printf("PufferLib Chess Console Demo\n");
  srand(time(NULL));

  const char *weights_path = "pufferlib/resources/chess/puffer_chess_weights.bin";
  Weights *weights = NULL;
  ChessNet *agent_net = NULL;
  
  // Try to load weights
  FILE *weight_file = fopen(weights_path, "rb");
  if (weight_file) {
    fclose(weight_file);
    weights = load_weights(weights_path, CHESS_NUM_WEIGHTS);
    if (!weights) {
      fprintf(stderr, "ERROR: Could not load weights from %s\n", weights_path);
      return 1;
    }
    printf("Loaded pre-trained weights from %s\n", weights_path);
  } else {
    printf("No pre-trained weights found, using random initialization\n");
    weights = (Weights*)calloc(1, sizeof(Weights) + CHESS_NUM_WEIGHTS*sizeof(float));
    weights->data = (float*)(weights + 1);
    weights->size = CHESS_NUM_WEIGHTS;
    weights->idx = 0;
    for (int i = 0; i < CHESS_NUM_WEIGHTS; i++) {
      weights->data[i] = ((float)rand() / (float)RAND_MAX - 0.5f) * 0.02f;
    }
  }
  agent_net = init_chessnet(weights, 2);
  global_agent_net = agent_net;

  CChess env = {0};
  env.max_depth = 500;
  allocate(&env);
  set_dual_agent_self_play_mode(&env, false);  // Agent vs Random
  c_reset(&env);

  printf("\nStarting Agent vs Random game...\n");
  printf("Commands: 'q' to quit, 'r' to reset, 's' to show board\n\n");

  int move_count = 0;
  while (!env.terminals[0] && move_count < 100) {
    auto *ctx = env.ctx;
    printf("Move %d - %s to move\n", move_count + 1, 
           ctx->board.to_move == C_WHITE ? "White" : "Black");
    
    compute_observation_with_perspective(&env, ctx);
    int action = 0;
    int agent_idx = (ctx->board.to_move == C_WHITE) ? 0 : 1;

    if (ctx->board.to_move == C_WHITE) {
      // Agent plays as White
      action = agent_select_action(agent_net, &env, agent_idx);
      printf("[AGENT] Selected action %d\n", action);
    } else {
      // Random plays as Black
      action = random_select_action(&env);
      printf("[RANDOM] Selected action %d\n", action);
    }

    env.actions[0] = action;  // FIXED: c_step always reads from actions[0]
    c_step(&env);
    
    // Record the move for console display
    if (env.log.last_move_from >= 0 && env.log.last_move_to >= 0) {
      int from_square = (int)env.log.last_move_from;
      int to_square = (int)env.log.last_move_to;
      int from_x = from_square % 8;
      int from_y = from_square / 8;
      int to_x = to_square % 8;
      int to_y = to_square / 8;
      
      char move_uci[6];
      snprintf(move_uci, sizeof(move_uci), "%c%d%c%d", 
               'a' + from_x, from_y + 1, 'a' + to_x, to_y + 1);
      
      printf("Move %d: %s played %s\n", move_count + 1,
             ctx->board.to_move == C_BLACK ? "White" : "Black", // to_move switched after move
             move_uci);
    }
    
    move_count++;

    // Show the board every 10 moves
    if (move_count % 10 == 0) {
      printf("\nBoard state after move %d:\n", move_count);
      // Print a simple ASCII board representation
      for (int rank = 7; rank >= 0; rank--) {
        printf("%d ", rank + 1);
        for (int file = 0; file < 8; file++) {
          const Piece *p = get_piece_const(&ctx->board, file, rank);
          char piece_char = '.';
          if (p && p->type != EMPTY) {
            switch (p->type) {
              case KING: piece_char = (p->color == C_WHITE) ? 'K' : 'k'; break;
              case QUEEN: piece_char = (p->color == C_WHITE) ? 'Q' : 'q'; break;
              case ROOK: piece_char = (p->color == C_WHITE) ? 'R' : 'r'; break;
              case BISHOP: piece_char = (p->color == C_WHITE) ? 'B' : 'b'; break;
              case KNIGHT: piece_char = (p->color == C_WHITE) ? 'N' : 'n'; break;
              case PAWN: piece_char = (p->color == C_WHITE) ? 'P' : 'p'; break;
              default: piece_char = '?'; break;
            }
          }
          printf("%c ", piece_char);
        }
        printf("\n");
      }
      printf("  a b c d e f g h\n\n");
    }
  }

  printf("\nGame ended after %d moves\n", move_count);
  if (env.terminals[0]) {
    printf("Game result: Terminal state reached\n");
  }

  // Cleanup
  if (agent_net) {
    free_chessnet(agent_net);
  }
  if (weights) {
    free(weights);
  }
  free_allocated(&env);
  
  return 0;
}

// PGN loading functionality
bool load_pgn_file(CChess *env, const char *filename) {
  FILE *file = fopen(filename, "r");
  if (!file) {
    printf("[Chess] Failed to open PGN file: %s\n", filename);
    return false;
  }
  
  char line[512];
  bool in_moves = false;
  
  // Reset to starting position
  c_reset(env);
  
  printf("[Chess] Loading PGN file: %s\n", filename);
  
  while (fgets(line, sizeof(line), file)) {
    // Skip header lines that start with [
    if (line[0] == '[') {
      continue;
    }
    
    // Empty line usually separates headers from moves
    if (line[0] == '\n' || line[0] == '\r') {
      in_moves = true;
      continue;
    }
    
    if (in_moves) {
      // Parse moves from the line
      char *token = strtok(line, " \t\n\r");
      while (token != NULL) {
        // Skip move numbers (e.g., "1.", "2.")
        if (strchr(token, '.') != NULL) {
          token = strtok(NULL, " \t\n\r");
          continue;
        }
        
        // Skip result markers
        if (strcmp(token, "*") == 0 || strcmp(token, "1-0") == 0 || 
            strcmp(token, "0-1") == 0 || strcmp(token, "1/2-1/2") == 0) {
          break;
        }
        
        // Try to apply the UCI move
        if (strlen(token) >= 4) {
          printf("[Chess] Applying move: %s\n", token);
          if (!apply_uci_move(&env->context, token)) {
            printf("[Chess] Failed to apply move: %s\n", token);
            fclose(file);
            return false;
          }
        }
        
        token = strtok(NULL, " \t\n\r");
      }
    }
  }
  
  fclose(file);
  printf("[Chess] PGN loaded successfully\n");
  return true;
}

// Game viewer state
struct GameViewer {
  std::vector<std::string> moves;
  int current_move;
  CChess env;
  bool game_loaded;
};

bool load_pgn_to_viewer(GameViewer *viewer, const char *filename) {
  FILE *file = fopen(filename, "r");
  if (!file) {
    printf("[Chess] Failed to open PGN file: %s\n", filename);
    return false;
  }
  
  viewer->moves.clear();
  viewer->current_move = 0;
  viewer->game_loaded = false;
  
  char line[512];
  bool in_moves = false;
  
  printf("[Chess] Loading PGN file: %s\n", filename);
  
  while (fgets(line, sizeof(line), file)) {
    if (line[0] == '[') continue;
    if (line[0] == '\n' || line[0] == '\r') {
      in_moves = true;
      continue;
    }
    
    if (in_moves) {
      char *token = strtok(line, " \t\n\r");
      while (token != NULL) {
        if (strchr(token, '.') != NULL) {
          token = strtok(NULL, " \t\n\r");
          continue;
        }
        if (strcmp(token, "*") == 0 || strcmp(token, "1-0") == 0 || 
            strcmp(token, "0-1") == 0 || strcmp(token, "1/2-1/2") == 0) {
          break;
        }
        if (strlen(token) >= 4) {
          viewer->moves.push_back(std::string(token));
        }
        token = strtok(NULL, " \t\n\r");
      }
    }
  }
  
  fclose(file);
  
  // Reset to starting position
  c_reset(&viewer->env);
  viewer->game_loaded = true;
  
  printf("[Chess] Loaded %zu moves\n", viewer->moves.size());
  return true;
}

void apply_moves_to_position(GameViewer *viewer, int up_to_move) {
  // Reset to starting position
  c_reset(&viewer->env);
  
  // Apply moves one by one to reach the desired position
  for (int i = 0; i < up_to_move && i < (int)viewer->moves.size(); i++) {
    const char* uci_move = viewer->moves[i].c_str();
    const char* player = (i % 2 == 0) ? "WHITE" : "BLACK";
    int action_id = uci_to_action_id(uci_move);
    
    printf("[VIEWER_DEBUG] Move %d (%s): PGN_UCI='%s' -> action_id=%d\n", 
           i+1, player, uci_move, action_id);
    
    // Print board state before move
    printf("[VIEWER_DEBUG] Board before move %d:\n", i+1);
    for (int rank = 7; rank >= 0; rank--) {
      printf("[VIEWER_DEBUG] ");
      for (int file = 0; file < 8; file++) {
        Piece* piece = get_piece(&viewer->env.context.board, file, rank);
        char piece_char = '.';
        if (piece && piece->type != EMPTY) {
          char piece_symbols[] = ".KQRBNP";
          piece_char = piece_symbols[piece->type];
          if (piece->color == C_BLACK) piece_char = std::tolower(piece_char);
        }
        printf("%c", piece_char);
      }
      printf(" %d\n", rank + 1);
    }
    printf("[VIEWER_DEBUG] abcdefgh\n");
    
    // Use apply_uci_move directly instead of going through c_step
    // This avoids the perspective conversion that c_step applies internally
    bool move_applied = apply_uci_move(&viewer->env.context, uci_move);
    
    if (move_applied) {
      // Print board state after move
      printf("[VIEWER_DEBUG] Board after applying PGN move '%s' (direct apply_uci_move):\n", 
             uci_move);
      for (int rank = 7; rank >= 0; rank--) {
        printf("[VIEWER_DEBUG] ");
        for (int file = 0; file < 8; file++) {
          Piece* piece = get_piece(&viewer->env.context.board, file, rank);
          char piece_char = '.';
          if (piece && piece->type != EMPTY) {
            char piece_symbols[] = ".KQRBNP";
            piece_char = piece_symbols[piece->type];
            if (piece->color == C_BLACK) piece_char = std::tolower(piece_char);
          }
          printf("%c", piece_char);
        }
        printf(" %d\n", rank + 1);
      }
      printf("[VIEWER_DEBUG] abcdefgh\n");
      printf("[VIEWER_DEBUG] -----------\n");
    } else {
      printf("[Chess] Warning: Failed to apply UCI move '%s' directly\n", uci_move);
    }
  }
}

std::string get_pgn_result(const std::string& filename) {
  FILE *file = fopen(filename.c_str(), "r");
  if (!file) return "*";
  
  char line[512];
  while (fgets(line, sizeof(line), file)) {
    if (strncmp(line, "[Result ", 8) == 0) {
      char *start = strchr(line, '"');
      if (start) {
        start++;
        char *end = strchr(start, '"');
        if (end) {
          *end = '\0';
          std::string result = start;
          fclose(file);
          return result;
        }
      }
    }
  }
  fclose(file);
  return "*";
}

std::vector<std::string> get_pgn_files() {
  std::vector<std::string> files;
  const char* dir_path = "resources/chess/training_logs/complete_games";
  
  DIR *dir = opendir(dir_path);
  if (!dir) return files;
  
  struct dirent *entry;
  while ((entry = readdir(dir)) != NULL) {
    std::string filename = entry->d_name;
    if (filename.size() > 4 && filename.substr(filename.size() - 4) == ".pgn") {
      files.push_back(std::string(dir_path) + "/" + filename);
    }
  }
  closedir(dir);
  
  std::sort(files.begin(), files.end());
  return files;
}

int game_browser() {
  const int screenWidth = 1200;
  const int screenHeight = 800;
  InitWindow(screenWidth, screenHeight, "Chess Game Browser");
  SetTargetFPS(60);
  
  ChessPieceTextures textures = load_piece_textures();
  GameViewer viewer = {};
  allocate(&viewer.env);
  
  std::vector<std::string> pgn_files = get_pgn_files();
  int selected_file = 0;
  int scroll_offset = 0;  // For scrolling when list is long
  bool show_file_list = true;
  
  while (!WindowShouldClose()) {
    // Input handling
    if (show_file_list) {
      // Use IsKeyDown for rapid scrolling when holding arrow keys
      if (IsKeyDown(KEY_UP) && selected_file > 0) {
        selected_file--;
        // Auto-scroll up if selection goes above visible area
        if (selected_file < scroll_offset) {
          scroll_offset = selected_file;
        }
      }
      if (IsKeyDown(KEY_DOWN) && selected_file < (int)pgn_files.size() - 1) {
        selected_file++;
        // Auto-scroll down if selection goes below visible area
        const int visible_lines = 30; // Approximate number of visible lines
        if (selected_file >= scroll_offset + visible_lines) {
          scroll_offset = selected_file - visible_lines + 1;
        }
      }
      // Page up/down for faster navigation
      if (IsKeyPressed(KEY_PAGE_UP)) {
        selected_file = std::max(0, selected_file - 10);
        scroll_offset = std::max(0, scroll_offset - 10);
      }
      if (IsKeyPressed(KEY_PAGE_DOWN)) {
        selected_file = std::min((int)pgn_files.size() - 1, selected_file + 10);
        const int visible_lines = 30;
        if (selected_file >= scroll_offset + visible_lines) {
          scroll_offset = selected_file - visible_lines + 1;
        }
      }
      if (IsKeyPressed(KEY_ENTER) && !pgn_files.empty()) {
        if (load_pgn_to_viewer(&viewer, pgn_files[selected_file].c_str())) {
          show_file_list = false;
        }
      }
      if (IsKeyPressed(KEY_ESCAPE)) {
        break;  // Exit the program
      }
    } else {
      // Use IsKeyDown for rapid navigation when holding arrow keys
      if (IsKeyDown(KEY_LEFT) && viewer.current_move > 0) {
        viewer.current_move--;
        apply_moves_to_position(&viewer, viewer.current_move);
      }
      if (IsKeyDown(KEY_RIGHT) && viewer.current_move < (int)viewer.moves.size()) {
        viewer.current_move++;
        apply_moves_to_position(&viewer, viewer.current_move);
      }
      if (IsKeyPressed(KEY_HOME)) {
        viewer.current_move = 0;
        apply_moves_to_position(&viewer, viewer.current_move);
      }
      if (IsKeyPressed(KEY_END)) {
        viewer.current_move = viewer.moves.size();
        apply_moves_to_position(&viewer, viewer.current_move);
      }
      if (IsKeyPressed(KEY_M)) {
        show_file_list = true;
      }
      if (IsKeyPressed(KEY_ESCAPE)) {
        break;  // Exit the program
      }
    }
    
    BeginDrawing();
    ClearBackground(RL_RAYWHITE);
    
    if (show_file_list) {
      DrawText("Chess Game Browser", 10, 10, 24, RL_BLACK);
      DrawText("UP/DOWN: select (hold for rapid), ENTER: load, ESC: exit", 10, 40, 16, RL_DARKGRAY);
      DrawText("PAGE UP/DOWN: jump 10 games", 10, 60, 16, RL_DARKGRAY);
      DrawText("Game Results: 1-0=White wins, 0-1=Black wins, 1/2-1/2=Draw, *=In progress/Unknown", 10, 80, 14, RL_DARKGRAY);
      
      // Calculate visible range
      const int visible_lines = 30;
      const int start_y = 110;
      const int line_height = 20;
      
      // Show scroll position info
      if (pgn_files.size() > visible_lines) {
        char scroll_info[128];
        sprintf(scroll_info, "Showing %d-%d of %d games (Page %d/%d)", 
                scroll_offset + 1, 
                std::min(scroll_offset + visible_lines, (int)pgn_files.size()),
                (int)pgn_files.size(),
                (scroll_offset / visible_lines) + 1,
                ((int)pgn_files.size() - 1) / visible_lines + 1);
        DrawText(scroll_info, 10, start_y - 25, 14, RL_DARKBLUE);
      }
      
      // Draw visible files only
      int displayed_count = 0;
      for (int i = scroll_offset; i < (int)pgn_files.size() && displayed_count < visible_lines; i++) {
        Color color = (i == selected_file) ? RL_BLUE : RL_BLACK;
        Color bg_color = (i == selected_file) ? RL_LIGHTGRAY : RL_RAYWHITE;
        
        std::string filename = pgn_files[i].substr(pgn_files[i].find_last_of("/") + 1);
        std::string result = get_pgn_result(pgn_files[i]);
        std::string display_text = filename + " [" + result + "]";
        
        int y_pos = start_y + displayed_count * line_height;
        
        // Draw selection background
        if (i == selected_file) {
          DrawRectangle(5, y_pos - 2, 1190, line_height, bg_color);
        }
        
        DrawText(display_text.c_str(), 10, y_pos, 16, color);
        displayed_count++;
      }
      
      if (pgn_files.empty()) {
        DrawText("No PGN files found in resources/chess/training_logs/complete_games/", 10, 140, 16, RL_RED);
      }
    } else {
      // Draw chess board
      render_chess_board(&viewer.env, &textures);
      
      // Draw controls
      DrawText("Controls:", 850, 50, 20, RL_BLACK);
      DrawText("LEFT/RIGHT: Previous/Next move (hold for rapid)", 850, 80, 14, RL_DARKGRAY);
      DrawText("HOME/END: Start/End of game", 850, 100, 16, RL_DARKGRAY);
      DrawText("M: Back to file list", 850, 120, 16, RL_DARKGRAY);
      DrawText("ESC: Exit program", 850, 140, 16, RL_DARKGRAY);
      
      // Draw move info
      char move_info[128];
      sprintf(move_info, "Move: %d / %d", viewer.current_move, (int)viewer.moves.size());
      DrawText(move_info, 850, 180, 18, RL_BLACK);
      
      if (viewer.current_move > 0 && viewer.current_move <= (int)viewer.moves.size()) {
        const char* last_move = viewer.moves[viewer.current_move - 1].c_str();
        sprintf(move_info, "Last: %s", last_move);
        DrawText(move_info, 850, 200, 16, RL_DARKGRAY);
      }
      
      // Draw move list in standard chess notation format
      DrawText("Game Moves:", 850, 240, 18, RL_BLACK);
      const int moves_start_y = 270;
      const int move_height = 16;
      const int visible_move_lines = 25; // Show about 25 lines of moves
      
      // Calculate which move pairs to show (scroll to keep current move visible)
      int current_move_pair = (viewer.current_move - 1) / 2; // 0-based move pair number
      int scroll_start = std::max(0, current_move_pair - visible_move_lines / 2);
      int lines_shown = 0;
      
      // Draw moves in standard format: "1. e4 e5"
      for (int move_pair = scroll_start; move_pair < ((int)viewer.moves.size() + 1) / 2 && lines_shown < visible_move_lines; move_pair++) {
        int y_pos = moves_start_y + lines_shown * move_height;
        int white_move_idx = move_pair * 2;
        int black_move_idx = move_pair * 2 + 1;
        
        // Draw move number
        char move_num[16];
        sprintf(move_num, "%d.", move_pair + 1);
        
        // Highlight move number if current move is in this pair
        Color num_color = RL_DARKGRAY;
        if (viewer.current_move > 0 && (viewer.current_move - 1) / 2 == move_pair) {
          num_color = RL_BLUE;
        }
        DrawText(move_num, 850, y_pos, 14, num_color);
        
        // Draw white move (always exists for this pair)
        if (white_move_idx < (int)viewer.moves.size()) {
          const char* white_move = viewer.moves[white_move_idx].c_str();
          
          // Highlight current move
          Color white_color = RL_BLACK;
          if (white_move_idx == viewer.current_move - 1) {
            white_color = RL_RED;
            DrawRectangle(878, y_pos - 1, 70, 16, RL_BEIGE);
          } else if (white_move_idx < viewer.current_move - 1) {
            white_color = RL_DARKGRAY;
          }
          
          DrawText(white_move, 880, y_pos, 12, white_color);
        }
        
        // Draw black move (if it exists)
        if (black_move_idx < (int)viewer.moves.size()) {
          const char* black_move = viewer.moves[black_move_idx].c_str();
          
          // Highlight current move
          Color black_color = RL_BLACK;
          if (black_move_idx == viewer.current_move - 1) {
            black_color = RL_RED;
            DrawRectangle(958, y_pos - 1, 70, 16, RL_BEIGE);
          } else if (black_move_idx < viewer.current_move - 1) {
            black_color = RL_DARKGRAY;
          }
          
          DrawText(black_move, 960, y_pos, 12, black_color);
        }
        
        lines_shown++;
      }
      
      // Show scroll indicator if needed
      int total_move_pairs = ((int)viewer.moves.size() + 1) / 2;
      if (total_move_pairs > visible_move_lines) {
        char scroll_indicator[64];
        sprintf(scroll_indicator, "(Showing moves %d-%d of %d)", 
                scroll_start + 1,
                std::min(scroll_start + visible_move_lines, total_move_pairs),
                total_move_pairs);
        DrawText(scroll_indicator, 850, moves_start_y + visible_move_lines * move_height + 10, 12, RL_DARKBLUE);
      }
    }
    
    EndDrawing();
  }
  
  unload_piece_textures(&textures);
  CloseWindow();
  free_allocated(&viewer.env);
  return 0;
}

int main(int argc, char **argv) {
  srand(time(NULL));

  if (argc > 1 && strcmp(argv[1], "demo") == 0) {
    printf("Starting chess demo with neural network agent...\n");
    return demo();
  } else if (argc > 1 && strcmp(argv[1], "console") == 0) {
    printf("Starting console-only chess demo...\n");
    return demo_console();
  } else if (argc > 1 && strcmp(argv[1], "browser") == 0) {
    printf("Starting game browser...\n");
    return game_browser();
  } else {
    printf("Usage:\n");
    printf("  %s demo     - Interactive chess demo\n", argv[0]);
    printf("  %s console  - Console chess demo\n", argv[0]);
    printf("  %s browser  - Browse and view training games\n", argv[0]);
    printf("  %s          - Run performance test\n", argv[0]);
    printf("\nRunning performance test...\n");
    test_performance(10);
  }

  return 0;
}