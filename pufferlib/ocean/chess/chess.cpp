// chess.cpp - Graphical Chess Evaluation using Raylib
#include <time.h>
#include <math.h>
#include "chess.h"
#include "puffernet.h"
#include "stockfish_wrapper.h"
#include <cstdlib>
#include <cstdio>
#include <string>
#include <utility>
#include <cstring>
#include <unistd.h> // Added for access() to check executable presence
#include <algorithm>
#include <vector>
#include <sstream>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <unordered_map>
#include <climits>
#include <random> // For random moves when Stockfish is disabled
#include <filesystem> // Added for directory handling
#include <nlohmann/json.hpp> // Added for JSON handling

#ifdef __cplusplus
extern "C" {
#endif
#include "raylib.h" 
#ifdef __cplusplus
}
#endif

// Preserve raylib color constants before undefining macros that clash with our enum names
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

// Raylib defines macros WHITE, BLACK that clash with chess::Color constants used later with qualification.
#ifdef WHITE
#undef WHITE
#endif
#ifdef BLACK
#undef BLACK
#endif

namespace chess {

// GameOutcome is already defined in chess.h, use the existing one
GameOutcome last_game_outcome;

// -----------------------------------------------------------------------------
// ChessNet definition – matches the current ChessRecurrent class in torch.py
// Architecture: 1344 -> 512 -> 256 -> 256 (hidden) -> 4674 (policy) + value head
// -----------------------------------------------------------------------------

typedef struct ChessNet ChessNet;
struct ChessNet {
    int num_agents;
    
    // Board encoder: 1344 -> 512 -> 256
    Linear *board_enc1;    // 1344 -> 512
    ReLU   *board_relu1;
    Linear *board_enc2;    // 512 -> 256  
    ReLU   *board_relu2;
    
    // Combiner: 256 -> 256 (hidden_size)
    Linear *combiner;      // 256 -> 256
    ReLU   *comb_relu;
    
    // LSTM: input_size=256, hidden_size=256
    LSTM   *lstm;          // 256 -> 256
    
    // Policy head: 256 -> 4674
    Linear *policy_head;   // 256 -> 4674
    
    // Value head: 256 -> 128 -> 1
    Linear *value_head1;   // 256 -> 128
    ReLU   *value_relu;
    Linear *value_head2;   // 128 -> 1

    Multidiscrete *md;     // For action selection (4674 actions)
};

// Calculate total weights for current architecture with LSTM:
// board_encoder: (1344*512 + 512) + (512*256 + 256) = 688128 + 131328 = 819456
// combiner: (256*256 + 256) = 65792
// policy_head: (256*4674 + 4674) = 1201218
// value_head: (256*128 + 128) + (128*1 + 1) = 32897
// LSTM: 4 * ((256+256)*256 + 256) = 4 * 131328 = 525312
// Total: 2644675 (file has ~2,646,339 - close match)
#define CHESS_NUM_WEIGHTS 2646339

// Utility to mask invalid move logits before softmax sampling
static inline void mask_logits(float *logits, const float *legal, int size) {
    for (int i = 0; i < size; ++i) {
        if (legal[i] < 0.5f) logits[i] = -1e9f; // effectively -inf
    }
}

static ChessNet *init_chessnet(Weights *weights, int num_agents) {
    ChessNet *net = (ChessNet *)calloc(1, sizeof(ChessNet));
    net->num_agents = num_agents;

    // Board encoder: 1344 -> 512 -> 256
    net->board_enc1  = make_linear(weights, num_agents, 1344, 512);
    net->board_relu1 = make_relu(num_agents, 512);
    net->board_enc2  = make_linear(weights, num_agents, 512, 256);
    net->board_relu2 = make_relu(num_agents, 256);
    
    // Combiner: 256 -> 256 
    net->combiner    = make_linear(weights, num_agents, 256, 256);
    net->comb_relu   = make_relu(num_agents, 256);
    
    // LSTM: input_size=256, hidden_size=256
    net->lstm        = make_lstm(weights, num_agents, 256, 256);
    
    // Policy head: 256 -> 4674
    net->policy_head = make_linear(weights, num_agents, 256, 4674);
    
    // Value head: 256 -> 128 -> 1
    net->value_head1 = make_linear(weights, num_agents, 256, 128);
    net->value_relu  = make_relu(num_agents, 128);
    net->value_head2 = make_linear(weights, num_agents, 128, 1);

    // Use 4674 actions for multidiscrete (matching chess action space)
    int logit_sizes[1] = {4674};
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

// Forward pass – fills `actions` with the selected move for each agent.
static void forward_chessnet(ChessNet *net, float *observations, int *actions) {
    // Board encoder: 1344 -> 512 -> 256
    linear(net->board_enc1, observations);
    relu(net->board_relu1, net->board_enc1->output);
    linear(net->board_enc2, net->board_relu1->output);
    relu(net->board_relu2, net->board_enc2->output);
    
    // Combiner: 256 -> 256
    linear(net->combiner, net->board_relu2->output);
    relu(net->comb_relu, net->combiner->output);
    
    // LSTM: 256 -> 256
    lstm(net->lstm, net->comb_relu->output);
    
    // Policy head: 256 -> 4674 (using LSTM hidden state)
    linear(net->policy_head, net->lstm->state_h);
    
    // Mask illegal moves (observations[1344:1344+4674])
    const float *legal = observations + 1344;
    mask_logits(net->policy_head->output, legal, 4674);
    
    // Select action using softmax sampling for more natural play
    softmax_multidiscrete(net->md, net->policy_head->output, actions);
    
    // Value head (using LSTM hidden state)
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
    textures.wqueen = LoadTexture("resources/chess/wqueen.png");
    textures.wrook = LoadTexture("resources/chess/wrook.png");
    textures.wbishop = LoadTexture("resources/chess/wbishop.png");
    textures.wknight = LoadTexture("resources/chess/wknight.png");
    textures.wpawn = LoadTexture("resources/chess/wpawn.png");
    
    textures.bking = LoadTexture("resources/chess/bking.png");
    textures.bqueen = LoadTexture("resources/chess/bqueen.png");
    textures.brook = LoadTexture("resources/chess/brook.png");
    textures.bbishop = LoadTexture("resources/chess/bbishop.png");
    textures.bknight = LoadTexture("resources/chess/bknight.png");
    textures.bpawn = LoadTexture("resources/chess/bpawn.png");
    
    printf("Texture loading completed successfully.\n"); 
    
    return textures;
}

static void unload_piece_textures(ChessPieceTextures *textures) {
    UnloadTexture(textures->wking);
    UnloadTexture(textures->wqueen);
    UnloadTexture(textures->wrook);
    UnloadTexture(textures->wbishop);
    UnloadTexture(textures->wknight);
    UnloadTexture(textures->wpawn);
    UnloadTexture(textures->bking);
    UnloadTexture(textures->bqueen);
    UnloadTexture(textures->brook);
    UnloadTexture(textures->bbishop);
    UnloadTexture(textures->bknight);
    UnloadTexture(textures->bpawn);
}

static Texture2D get_piece_texture(const ChessPieceTextures *textures, Color color, PieceType type) {
    Texture2D empty_texture = {0};
    if (type == EMPTY) return empty_texture;
    
    if (color == 0) { // WHITE = 0
        switch (type) {
            case KING: return textures->wking;
            case QUEEN: return textures->wqueen;
            case ROOK: return textures->wrook;
            case BISHOP: return textures->wbishop;
            case KNIGHT: return textures->wknight;
            case PAWN: return textures->wpawn;
            case EMPTY: 
            default: return empty_texture;
        }
    } else {
        switch (type) {
            case KING: return textures->bking;
            case QUEEN: return textures->bqueen;
            case ROOK: return textures->brook;
            case BISHOP: return textures->bbishop;
            case KNIGHT: return textures->bknight;
            case PAWN: return textures->bpawn;
            case EMPTY:
            default: return empty_texture;
        }
    }
}

// Board rendering constants
const int BOARD_SIZE = 512;
const int SQUARE_SIZE = BOARD_SIZE / 8;
const int BOARD_OFFSET_X = 50;
const int BOARD_OFFSET_Y = 70;

// Constants for rendering
const int WINDOW_WIDTH = 900;
const int WINDOW_HEIGHT = 700;

// Forward declaration of GameLogger class
class GameLogger;
GameLogger* global_game_logger = nullptr;
std::vector<std::string> available_games;
int selected_game_index = 0;
bool show_game_list = false;
bool replay_mode_active = false;
bool auto_play_replay = false;
int auto_play_delay = 500;
bool game_paused = false;
bool show_bestmove = false;
std::vector<std::string> game_moves;
int panel_offset_x = 0;
int panel_offset_y = 0;
int last_panel_offset_x = 0;
int last_panel_offset_y = 0;
int session_wins = 0;
int session_losses = 0;
int session_draws = 0;

// Promotion selection state
bool show_promotion_selection = false;
int promotion_from_x = -1;
int promotion_from_y = -1;
int promotion_to_x = -1;
int promotion_to_y = -1;
PieceType selected_promotion = QUEEN;

// Global pointer to the current CChess environment instance
CChess* global_env_ptr = nullptr;

// UI constants
const int DETAILS_START_X = 650;
const int DETAILS_START_Y = 100;

// Game mode enumeration
enum GameMode {
    GM_PLAYER_STOCKFISH = 0,
    GM_PLAYER_RANDOM,
    GM_AGENT_STOCKFISH,
    GM_AGENT_AGENT,
    GM_AGENT_RANDOM,
    GM_RANDOM_RANDOM,
    GM_RANDOM_AGENT,
    GM_GAME_REPLAY,
    GM_COUNT
};

// Game mode names
const char* GAME_MODE_NAMES[] = {
    "Player vs Stockfish",
    "Player vs Random",
    "Agent vs Stockfish",
    "Agent vs Agent",
    "Agent vs Random",
    "Random vs Random",
    "Random vs Agent",
    "Game Replay"
};

// Game logging structures for reading training logs
struct GameMove {
    int move_number;
    int action_id;
    std::string algebraic_notation;
};

struct GameLogEntry {
    std::string filename;
    std::string timestamp;
    std::string outcome;
    std::string draw_reason; // Added for specific draw types
    int total_moves;
    std::vector<GameMove> moves;
};

// Game replay structures
struct GameReplay {
    bool is_active = false;
    const GameLogEntry* current_game = nullptr;
    int current_move_index = 0;
    
    bool start_replay(const GameLogEntry* game) {
        if (!game) return false;
        current_game = game;
        current_move_index = 0;
        is_active = true;
        return true;
    }
    
    bool next_move() {
        if (!is_active || !current_game) return false;
        if (current_move_index >= current_game->total_moves - 1) return false;
        current_move_index++;
        return true;
    }
    
    bool prev_move() {
        if (!is_active || !current_game) return false;
        if (current_move_index <= 0) return false;
        current_move_index--;
        return true;
    }
    
    bool jump_to_move(int move_index) {
        if (!is_active || !current_game) return false;
        if (move_index < 0 || move_index >= current_game->total_moves) return false;
        current_move_index = move_index;
        return true;
    }
    
    const GameMove* get_current_move() const {
        if (!is_active || !current_game) return nullptr;
        if (current_move_index >= (int)current_game->moves.size()) return nullptr;
        return &current_game->moves[current_move_index];
    }
};

GameReplay current_replay;

// Session statistics
struct PlayerStats {
    int wins = 0;
    int losses = 0;
    int draws = 0;
    int games = 0;
    
    float win_rate() const {
        return games > 0 ? (float)wins / games : 0.0f;
    }
    
    void add_win() { wins++; games++; }
    void add_loss() { losses++; games++; }
    void add_draw() { draws++; games++; }
    void reset() { wins = losses = draws = games = 0; }
};

struct SessionStats {
    int total_games = 0;
    int total_wins = 0;
    int total_losses = 0;
    int total_draws = 0;
    
    PlayerStats agent_stats;
    PlayerStats human_stats;
    PlayerStats white_stats;
    PlayerStats black_stats;
    
    void reset() {
        total_games = total_wins = total_losses = total_draws = 0;
        agent_stats.reset();
        human_stats.reset();
        white_stats.reset();
        black_stats.reset();
    }
    
    void print_summary(GameMode mode) const {
        printf("=== Session Statistics Summary ===\n");
        printf("Total Games: %d\n", total_games);
        printf("White Wins: %d, Black Wins: %d, Draws: %d\n", total_wins, total_losses, total_draws);
        if (agent_stats.games > 0) {
            printf("Agent: %.1f%% win rate (%d/%d/%d)\n", 
                   agent_stats.win_rate() * 100, agent_stats.wins, agent_stats.losses, agent_stats.draws);
        }
        if (human_stats.games > 0) {
            printf("Human: %.1f%% win rate (%d/%d/%d)\n", 
                   human_stats.win_rate() * 100, human_stats.wins, human_stats.losses, human_stats.draws);
        }
    }
    
    void save_to_file(GameMode mode, const std::string& filename = "chess_eval_stats.txt") const {
        std::ofstream file(filename);
        if (file.is_open()) {
            file << "Chess Evaluation Statistics\n";
            file << "Mode: " << GAME_MODE_NAMES[mode] << "\n";
            file << "Total Games: " << total_games << "\n";
            file << "White Wins: " << total_wins << ", Black Wins: " << total_losses << ", Draws: " << total_draws << "\n";
            if (agent_stats.games > 0) {
                file << "Agent: " << agent_stats.win_rate() * 100 << "% win rate (" 
                     << agent_stats.wins << "/" << agent_stats.losses << "/" << agent_stats.draws << ")\n";
            }
            if (human_stats.games > 0) {
                file << "Human: " << human_stats.win_rate() * 100 << "% win rate (" 
                     << human_stats.wins << "/" << human_stats.losses << "/" << human_stats.draws << ")\n";
            }
            file.close();
            printf("Statistics saved to %s\n", filename.c_str());
        }
    }
};

SessionStats session_stats;

// Forward declarations for functions
void load_available_games();
void render_game_list_screen();
void render_game_replay_screen(CChess* env, ChessPieceTextures* textures);
void handle_game_list_input();
void handle_game_replay_input();
void render_chess_board(CChess* env, ChessPieceTextures* textures);
void draw_side_panel(CChess* env, ChessPieceTextures* textures, GameMode mode, int elo_setting, ChessNet* white_net, ChessNet* black_net);
void apply_replay_moves_to_board(ChessBoard& board, const GameReplay& replay);
void update_session_stats(GameMode mode, bool white_won, bool black_won, bool is_draw);
int agent_select_action(ChessNet* net, CChess* env);
int agent_select_action_dual(ChessNet* net, CChess* env, int agent_index);
int random_select_action(ChessContext* ctx);
std::string move_to_uci(const Move& move);
void render_promotion_selection(ChessPieceTextures* textures);
void handle_promotion_selection();
bool is_promotion_move(const ChessBoard& board, int from_x, int from_y, int to_x, int to_y);

// To convert between JSON and GameLogEntry
void to_json(nlohmann::json& j, const GameMove& m) {
    j = nlohmann::json{{"move_number", m.move_number}, {"action_id", m.action_id}, {"algebraic_notation", m.algebraic_notation}};
}

void from_json(const nlohmann::json& j, GameMove& m) {
    j.at("move_number").get_to(m.move_number);
    j.at("action_id").get_to(m.action_id);
    j.at("algebraic_notation").get_to(m.algebraic_notation);
}

void to_json(nlohmann::json& j, const GameLogEntry& e) {
    j = nlohmann::json{{"filename", e.filename}, {"timestamp", e.timestamp}, {"outcome", e.outcome}, 
                       {"draw_reason", e.draw_reason}, {"total_moves", e.total_moves}, {"moves", e.moves}};
}

void from_json(const nlohmann::json& j, GameLogEntry& e) {
    j.at("filename").get_to(e.filename);
    j.at("timestamp").get_to(e.timestamp);
    j.at("outcome").get_to(e.outcome);
    if (j.contains("draw_reason")) {
        j.at("draw_reason").get_to(e.draw_reason);
    }
    j.at("total_moves").get_to(e.total_moves);
    j.at("moves").get_to(e.moves);
}

class GameLogger {
private:
    std::vector<GameLogEntry> games;
    std::string log_directory;

public:
    GameLogger(const std::string& directory) : log_directory(directory) {
        // Ensure log directory exists
        if (!std::filesystem::exists(log_directory)) {
            std::filesystem::create_directories(log_directory);
        }
    }

    void load_games_from_directory() {
        games.clear();
        for (const auto& entry : std::filesystem::directory_iterator(log_directory)) {
            if (entry.is_regular_file() && entry.path().extension() == ".json") {
                try {
                    std::ifstream file(entry.path());
                    nlohmann::json j;
                    file >> j;
                    games.push_back(j.get<GameLogEntry>());
                } catch (const nlohmann::json::parse_error& e) {
                    std::cerr << "Error parsing game log file " << entry.path() << ": " << e.what() << std::endl;
                }
            }
        }
    }

    void save_game(const GameLogEntry& entry) {
        std::string filename = log_directory + "/" + entry.filename + ".json";
        std::ofstream file(filename);
        if (file.is_open()) {
            nlohmann::json j = entry;
            file << std::setw(4) << j << std::endl;
            file.close();
        } else {
            std::cerr << "Error saving game log to " << filename << std::endl;
        }
    }

    const std::vector<GameLogEntry>& get_games() const { return games; }

    // Get game by filename
    const GameLogEntry* get_game_by_filename(const std::string& filename) const {
        for (const auto& game : games) {
            if (game.filename == filename) {
                return &game;
            }
        }
        return nullptr;
    }
};

} // namespace chess

// Function implementations
void chess::load_available_games() {
    available_games.clear();
    if (global_game_logger) {
        global_game_logger->load_games_from_directory();
        const auto& games = global_game_logger->get_games();
        for (size_t i = 0; i < games.size(); ++i) {
            char buffer[256];
            std::string outcome_display;
            
            // Convert outcome to clearer display format
            if (games[i].outcome == "win") {
                outcome_display = "White Win";
            } else if (games[i].outcome == "loss") {
                outcome_display = "Black Win";
            } else if (games[i].outcome == "draw") {
                // Show specific draw type if available
                if (!games[i].draw_reason.empty()) {
                    // Capitalize first letter of draw reason
                    std::string reason = games[i].draw_reason;
                    if (!reason.empty()) {
                        reason[0] = std::toupper(reason[0]);
                    }
                    outcome_display = "Draw (" + reason + ")";
                } else {
                    outcome_display = "Draw";
                }
            } else {
                outcome_display = games[i].outcome;
            }
            
            snprintf(buffer, sizeof(buffer), "Game %d: %s (%d moves) - %s", 
                    (int)i + 1, outcome_display.c_str(), games[i].total_moves, games[i].timestamp.c_str());
            available_games.push_back(std::string(buffer));
        }
    }
    printf("Loaded %d available games\n", (int)available_games.size());
}

void chess::render_game_list_screen() {
    ClearBackground(RL_RAYWHITE);
    
    DrawText("Game Replay - Select Game", 50, 20, 24, RL_BLACK);
    DrawText(TextFormat("Games available: %d", (int)available_games.size()), 50, 50, 16, RL_DARKGRAY);
    
    if (available_games.empty()) {
        DrawText("No games found in training logs directory", 50, 100, 18, RL_RED);
        DrawText("Press B to return to menu", 50, 130, 16, RL_DARKGRAY);
        return;
    }
    
    // Calculate scrolling
    const int GAMES_PER_PAGE = 20;
    const int start_y = 80;
    const int line_height = 20;
    
    int start_index = std::max(0, selected_game_index - GAMES_PER_PAGE / 2);
    int end_index = std::min((int)available_games.size(), start_index + GAMES_PER_PAGE);
    
    // Draw game list
    for (int i = start_index; i < end_index; ++i) {
        ::Color color = (i == selected_game_index) ? RL_RED : RL_BLACK;
        int y_pos = start_y + (i - start_index) * line_height;
        
        // Truncate long game descriptions
        std::string display_text = available_games[i];
        if (display_text.length() > 80) {
            display_text = display_text.substr(0, 77) + "...";
        }
        
        DrawText(display_text.c_str(), 50, y_pos, 14, color);
    }
    
    // Bottom navigation instructions
    DrawText("UP/DOWN: Navigate  ENTER: Select  B: Back", 50, WINDOW_HEIGHT - 40, 16, RL_DARKGRAY);
}

void chess::render_game_replay_screen(CChess* env, ChessPieceTextures* textures) {
    ClearBackground(RL_RAYWHITE);
    
    if (!current_replay.is_active) {
        DrawText("No game loaded for replay", 50, 300, 24, RL_RED);
        return;
    }
    
    // Apply moves from the replay to show the correct board position
    auto* ctx = (ChessContext*)env->context;
    if (ctx) {
        apply_replay_moves_to_board(ctx->board, current_replay);
    }
    
    // Render the chess board with the current replay position
    render_chess_board(env, textures);
    
    // Replay information panel
    int info_x = BOARD_OFFSET_X + BOARD_SIZE + 30;
    int info_y = BOARD_OFFSET_Y;
    
    DrawText("GAME REPLAY MODE", info_x, info_y, 20, RL_BLUE);
    info_y += 30;
    
    DrawText(TextFormat("Move: %d/%d", current_replay.current_move_index + 1, current_replay.current_game->total_moves), 
             info_x, info_y, 16, RL_BLACK);
    info_y += 25;
    
    const auto* current_move = current_replay.get_current_move();
    if (current_move) {
        DrawText(TextFormat("Action: %d", current_move->action_id), info_x, info_y, 14, RL_BLACK);
        info_y += 20;
        DrawText(TextFormat("UCI: %s", current_move->algebraic_notation.c_str()), info_x, info_y, 14, RL_BLACK);
        info_y += 25;
    }
    
    // Controls
    DrawText("CONTROLS:", info_x, info_y, 16, RL_DARKBLUE);
    info_y += 20;
    DrawText("LEFT/RIGHT: Step moves", info_x, info_y, 13, RL_DARKGRAY);
    info_y += 16;
    DrawText("HOME/END: Jump start/end", info_x, info_y, 13, RL_DARKGRAY);
    info_y += 16;
    DrawText("SPACE: Auto-play toggle", info_x, info_y, 13, RL_DARKGRAY);
    info_y += 16;
    DrawText("B: Back to game list", info_x, info_y, 13, RL_DARKGRAY);
    
    // Progress bar
    if (current_replay.current_game && current_replay.current_game->total_moves > 0) {
        int bar_y = WINDOW_HEIGHT - 25;
        int bar_width = WINDOW_WIDTH - 20;
        int bar_height = 15;
        
        DrawText(TextFormat("Progress: %d/%d moves", current_replay.current_move_index + 1, current_replay.current_game->total_moves), 
                 10, bar_y - 20, 14, RL_BLACK);
        
        DrawRectangle(10, bar_y, bar_width, bar_height, RL_LIGHTGRAY);
        
        float progress = (float)(current_replay.current_move_index + 1) / current_replay.current_game->total_moves;
        int progress_width = (int)(bar_width * progress);
        
        DrawRectangle(10, bar_y, progress_width, bar_height, RL_BLUE);
        DrawRectangleLines(10, bar_y, bar_width, bar_height, RL_BLACK);
    }
}

void chess::handle_game_list_input() {
    // Handle scrolling
    if (IsKeyPressed(KEY_UP)) {
        selected_game_index = std::max(0, selected_game_index - 1);
    } else if (IsKeyPressed(KEY_DOWN)) {
        selected_game_index = std::min(static_cast<int>(available_games.size()) - 1, selected_game_index + 1);
    }
    
    if (IsKeyPressed(KEY_ENTER)) {
        if (selected_game_index >= 0 && selected_game_index < static_cast<int>(available_games.size())) {
            if (global_game_logger) {
                const auto& games = global_game_logger->get_games();
                if (selected_game_index < (int)games.size()) {
                    current_replay.start_replay(&games[selected_game_index]);
                    show_game_list = false;
                    replay_mode_active = true;
                    printf("Loaded game for replay: %s\n", games[selected_game_index].timestamp.c_str());
                }
            }
        }
    }
    
    if (IsKeyPressed(KEY_B)) {
        show_game_list = false;
    }
}

void chess::handle_game_replay_input() {
    if (!current_replay.is_active) return;
    
    // Navigation controls
    if (IsKeyPressed(KEY_RIGHT) || IsKeyPressed(KEY_D)) {
        if (current_replay.next_move()) {
            printf("Stepped forward to move %d\n", current_replay.current_move_index + 1);
        }
    } else if (IsKeyPressed(KEY_LEFT) || IsKeyPressed(KEY_A)) {
        if (current_replay.prev_move()) {
            printf("Stepped backward to move %d\n", current_replay.current_move_index + 1);
        }
    }
    
    if (IsKeyPressed(KEY_HOME)) {
        current_replay.jump_to_move(0);
        printf("Jumped to start of game\n");
    }
    
    if (IsKeyPressed(KEY_END)) {
        current_replay.jump_to_move(current_replay.current_game->total_moves - 1);
        printf("Jumped to end of game\n");
    }
    
    if (IsKeyPressed(KEY_B)) {
        replay_mode_active = false;
        show_game_list = true;
    }
    
    // Auto-play controls
    if (IsKeyPressed(KEY_SPACE)) {
        auto_play_replay = !auto_play_replay;
        printf("Auto-play %s\n", auto_play_replay ? "ON" : "OFF");
    }
    
    // Auto-play logic
    if (auto_play_replay) {
        static auto last_auto_step = std::chrono::steady_clock::now();
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_auto_step);
        
        if (elapsed.count() >= auto_play_delay) {
            if (!current_replay.next_move()) {
                auto_play_replay = false; // Stop at end of game
                printf("Reached end of game, auto-play stopped\n");
            }
            last_auto_step = now;
        }
    }
}

void chess::apply_replay_moves_to_board(ChessBoard& board, const GameReplay& replay) {
    if (!replay.is_active || !replay.current_game) return;
    
    // Reset board to starting position
    board.reset();
    
    // Apply moves up to current position
    for (int i = 0; i <= replay.current_move_index && i < (int)replay.current_game->moves.size(); ++i) {
        const auto& game_move = replay.current_game->moves[i];
        
        // Convert action ID to move using the current board state
        Move move = action_to_move_lookup(game_move.action_id, board);
        
        // Validate and apply the move
        if (move.from.x >= 0 && move.from.y >= 0 && move.to.x >= 0 && move.to.y >= 0) {
            // Apply the move to the board
            bool applied = board.apply_move(move);
            if (!applied) {
                printf("[REPLAY ERROR] Failed to apply move %d: action %d\n", i + 1, game_move.action_id);
                break;
            }
        } else {
            printf("[REPLAY ERROR] Invalid move coordinates for action %d at move %d\n", 
                   game_move.action_id, i + 1);
            break;
        }
    }
}

void chess::update_session_stats(GameMode mode, bool white_won, bool black_won, bool is_draw) {
    session_stats.total_games++;
    
    if (white_won) {
        session_stats.total_wins++;
        session_stats.white_stats.add_win();
        session_stats.black_stats.add_loss();
        session_wins++;
    } else if (black_won) {
        session_stats.total_losses++;
        session_stats.white_stats.add_loss();
        session_stats.black_stats.add_win();
        session_losses++;
    } else if (is_draw) {
        session_stats.total_draws++;
        session_stats.white_stats.add_draw();
        session_stats.black_stats.add_draw();
        session_draws++;
    }
    
    // Update player-specific stats based on game mode
    switch (mode) {
        case GM_PLAYER_STOCKFISH:
        case GM_PLAYER_RANDOM:
            // Human plays white
            if (white_won) session_stats.human_stats.add_win();
            else if (black_won) session_stats.human_stats.add_loss();
            else if (is_draw) session_stats.human_stats.add_draw();
            break;
            
        case GM_AGENT_STOCKFISH:
        case GM_AGENT_AGENT:
        case GM_AGENT_RANDOM:
            // Agent plays white
            if (white_won) session_stats.agent_stats.add_win();
            else if (black_won) session_stats.agent_stats.add_loss();
            else if (is_draw) session_stats.agent_stats.add_draw();
            break;
            
        case GM_RANDOM_AGENT:
            // Agent plays black
            if (black_won) session_stats.agent_stats.add_win();
            else if (white_won) session_stats.agent_stats.add_loss();
            else if (is_draw) session_stats.agent_stats.add_draw();
            break;
            
        default:
            break;
    }
}

int chess::agent_select_action(ChessNet* net, CChess* env) {
    if (!net || !env) return 0;
    
    // Use the neural network to select an action
    int action;
    forward_chessnet(net, env->observations, &action);
    return action;
}

int chess::agent_select_action_dual(ChessNet* net, CChess* env, int agent_index) {
    if (!net || !env) return 0;
    
    // In dual agent mode, observations are laid out as [agent0_obs, agent1_obs]
    // Each agent has 6018 observation values
    float* agent_observations = env->observations + (agent_index * 6018);
    
    // Use the neural network to select an action
    int action;
    forward_chessnet(net, agent_observations, &action);
    return action;
}

int chess::random_select_action(ChessContext* ctx) {
    if (!ctx) return 0;
    
    const auto& legal_moves = ctx->board.legal_moves();
    if (legal_moves.empty()) return 0;
    
    // Select a random legal move
    std::uniform_int_distribution<int> dist(0, legal_moves.size() - 1);
    int move_index = dist(ctx->rng);
    
    return ChessBoard::move_to_action(legal_moves[move_index]);
}

bool chess::is_promotion_move(const chess::ChessBoard& board, int from_x, int from_y, int to_x, int to_y) {
    chess::Square from_pos{(int8_t)from_x, (int8_t)from_y};
    const chess::Piece &piece = board.at(from_pos);
    
    // Check if it's a pawn move
    if (piece.type != chess::PAWN) return false;
    
    // Check if it's moving to the promotion rank
    if (piece.color == chess::WHITE && to_y == 7) return true;
    if (piece.color == chess::BLACK && to_y == 0) return true;
    
    return false;
}

void chess::render_promotion_selection(ChessPieceTextures* textures) {
    if (!show_promotion_selection) return;
    
    // Draw semi-transparent overlay
    DrawRectangle(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT, ColorAlpha(RL_BLACK, 0.6f));
    
    // Draw promotion selection dialog
    int dialog_width = 320;
    int dialog_height = 160;
    int dialog_x = (WINDOW_WIDTH - dialog_width) / 2;
    int dialog_y = (WINDOW_HEIGHT - dialog_height) / 2;
    
    DrawRectangle(dialog_x, dialog_y, dialog_width, dialog_height, RL_LIGHTGRAY);
    DrawRectangleLines(dialog_x, dialog_y, dialog_width, dialog_height, RL_DARKGRAY);
    
    // Draw title
    DrawText("Choose Promotion Piece", dialog_x + 20, dialog_y + 20, 20, RL_DARKGRAY);
    
    // Draw promotion piece options
    const int piece_size = 64;
    const int piece_spacing = 70;
    const int start_x = dialog_x + 20;
    const int start_y = dialog_y + 60;
    
    chess::PieceType promotion_options[] = {chess::QUEEN, chess::ROOK, chess::BISHOP, chess::KNIGHT};
    const char* promotion_names[] = {"Queen", "Rook", "Bishop", "Knight"};
    
    for (int i = 0; i < 4; i++) {
        int piece_x = start_x + i * piece_spacing;
        int piece_y = start_y;
        
        // Highlight selected piece
        if (selected_promotion == promotion_options[i]) {
            DrawRectangle(piece_x - 2, piece_y - 2, piece_size + 4, piece_size + 4, RL_BLUE);
        }
        
        // Draw piece texture
        Texture2D texture = get_piece_texture(textures, chess::WHITE, promotion_options[i]);
        if (texture.id > 0) {
            Rectangle source = { 0.0f, 0.0f, (float)texture.width, (float)texture.height };
            Rectangle dest = { 
                (float)piece_x, 
                (float)piece_y, 
                (float)piece_size, 
                (float)piece_size 
            };
            Vector2 origin = { 0.0f, 0.0f };
            DrawTexturePro(texture, source, dest, origin, 0.0f, RL_WHITE);
        }
        
        // Draw piece name
        DrawText(promotion_names[i], piece_x + 5, piece_y + piece_size + 5, 12, RL_DARKGRAY);
        
        // Draw hotkey
        DrawText(TextFormat("%d", i + 1), piece_x + piece_size - 15, piece_y - 15, 14, RL_DARKGRAY);
    }
    
    // Draw instructions
    DrawText("Click piece or press 1-4 to select, Enter to confirm", dialog_x + 20, dialog_y + dialog_height - 30, 12, RL_DARKGRAY);
}

void chess::handle_promotion_selection() {
    if (!show_promotion_selection) return;
    
    // Handle keyboard input
    if (IsKeyPressed(KEY_ONE)) selected_promotion = chess::QUEEN;
    if (IsKeyPressed(KEY_TWO)) selected_promotion = chess::ROOK;
    if (IsKeyPressed(KEY_THREE)) selected_promotion = chess::BISHOP;
    if (IsKeyPressed(KEY_FOUR)) selected_promotion = chess::KNIGHT;
    
    // Handle mouse input
    if (IsMouseButtonPressed(MOUSE_LEFT_BUTTON)) {
        Vector2 mouse_pos = GetMousePosition();
        
        int dialog_width = 320;
        int dialog_height = 160;
        int dialog_x = (WINDOW_WIDTH - dialog_width) / 2;
        int dialog_y = (WINDOW_HEIGHT - dialog_height) / 2;
        
        const int piece_size = 64;
        const int piece_spacing = 70;
        const int start_x = dialog_x + 20;
        const int start_y = dialog_y + 60;
        
        chess::PieceType promotion_options[] = {chess::QUEEN, chess::ROOK, chess::BISHOP, chess::KNIGHT};
        
        for (int i = 0; i < 4; i++) {
            int piece_x = start_x + i * piece_spacing;
            int piece_y = start_y;
            
            if (mouse_pos.x >= piece_x && mouse_pos.x <= piece_x + piece_size &&
                mouse_pos.y >= piece_y && mouse_pos.y <= piece_y + piece_size) {
                selected_promotion = promotion_options[i];
                break;
            }
        }
    }
    
    // Confirm selection
    if (IsKeyPressed(KEY_ENTER) || IsKeyPressed(KEY_SPACE)) {
        show_promotion_selection = false;
        
        // Apply the promoted move
        if (promotion_from_x != -1 && promotion_from_y != -1 && 
            promotion_to_x != -1 && promotion_to_y != -1) {
            
            chess::Square from{(int8_t)promotion_from_x, (int8_t)promotion_from_y};
            chess::Square to{(int8_t)promotion_to_x, (int8_t)promotion_to_y};
            
            // Need to get the piece that is moving for the Move struct
            chess::Piece moving_piece = ((ChessContext*)global_env_ptr->context)->board.at(from);
            
            chess::Move promoted_move = {from, to, moving_piece, selected_promotion};
            
            global_env_ptr->actions[0] = chess::ChessBoard::move_to_action(promoted_move);
            
            // Record move before applying
            game_moves.push_back(move_to_uci(promoted_move));
            
            c_step(global_env_ptr); // Execute the promotion move
            
            // Reset promotion state
            promotion_from_x = -1;
            promotion_from_y = -1;
            promotion_to_x = -1;
            promotion_to_y = -1;
            selected_promotion = QUEEN; // Default back to queen
        }
    }
}

std::string chess::move_to_uci(const Move& move) {
    if (move.from.x < 0 || move.from.y < 0 || move.to.x < 0 || move.to.y < 0) {
        return "0000";
    }
    
    char from_file = 'a' + move.from.x;
    char from_rank = '1' + move.from.y;
    char to_file = 'a' + move.to.x;
    char to_rank = '1' + move.to.y;
    
    std::string uci = "";
    uci += from_file;
    uci += from_rank;
    uci += to_file;
    uci += to_rank;
    
    // Add promotion piece if applicable
    if (move.promotion != EMPTY) {
        switch (move.promotion) {
            case QUEEN: uci += "q"; break;
            case ROOK: uci += "r"; break;
            case BISHOP: uci += "b"; break;
            case KNIGHT: uci += "n"; break;
            default: break;
        }
    }
    
    return uci;
}

void chess::render_chess_board(CChess* env, ChessPieceTextures* textures) {
    if (!env || !textures) return;
    
    auto* ctx = (ChessContext*)env->context;
    if (!ctx) return;
    
    // Draw board squares
    for (int y = 0; y < 8; ++y) {
        for (int x = 0; x < 8; ++x) {
            int screen_x = BOARD_OFFSET_X + x * SQUARE_SIZE;
            int screen_y = BOARD_OFFSET_Y + (7 - y) * SQUARE_SIZE;
            
            ::Color square_color = ((x + y) % 2 == 0) ? RL_BEIGE : RL_BROWN;
            DrawRectangle(screen_x, screen_y, SQUARE_SIZE, SQUARE_SIZE, square_color);
            
            // Draw piece if present
            Square pos{int8_t(x), int8_t(y)};
            const Piece& piece = ctx->board.at(pos);
            
            if (piece.type != EMPTY) {
                Texture2D piece_texture = get_piece_texture(textures, piece.color, piece.type);
                if (piece_texture.id != 0) {
                    // Scale the piece texture to fit the square
                    Rectangle source = { 0.0f, 0.0f, (float)piece_texture.width, (float)piece_texture.height };
                    Rectangle dest = { 
                        (float)screen_x, 
                        (float)screen_y, 
                        (float)SQUARE_SIZE, 
                        (float)SQUARE_SIZE 
                    };
                    Vector2 origin = { 0.0f, 0.0f };
                    
                    DrawTexturePro(piece_texture, source, dest, origin, 0.0f, RL_WHITE);
                } else {
                    // Fallback to text if texture not available
                    const char* piece_chars = " KQRBNP";
                    char piece_char = piece_chars[piece.type];
                    if (piece.color == BLACK) {
                        piece_char = tolower(piece_char);
                    }
                    DrawText(TextFormat("%c", piece_char), screen_x + 20, screen_y + 20, 24, 
                            piece.color == WHITE ? RL_WHITE : RL_BLACK);
                }
            }
        }
    }
    
    // Draw board border
    DrawRectangleLines(BOARD_OFFSET_X - 2, BOARD_OFFSET_Y - 2, BOARD_SIZE + 4, BOARD_SIZE + 4, RL_BLACK);
    
    // Draw coordinates
    for (int i = 0; i < 8; ++i) {
        // Files (a-h)
        char file = 'a' + i;
        DrawText(TextFormat("%c", file), BOARD_OFFSET_X + i * SQUARE_SIZE + 30, BOARD_OFFSET_Y + BOARD_SIZE + 5, 16, RL_BLACK);
        
        // Ranks (1-8)
        char rank = '1' + i;
        DrawText(TextFormat("%c", rank), BOARD_OFFSET_X - 20, BOARD_OFFSET_Y + (7 - i) * SQUARE_SIZE + 30, 16, RL_BLACK);
    }
}

void chess::draw_side_panel(CChess* env, ChessPieceTextures* textures, GameMode mode, int elo_setting, ChessNet* white_net, ChessNet* black_net) {
    if (!env) return;
    
    auto* ctx = (ChessContext*)env->context;
    if (!ctx) return;
    
    int panel_x = BOARD_OFFSET_X + BOARD_SIZE + 30 + panel_offset_x;
    int panel_y = BOARD_OFFSET_Y + panel_offset_y;
    
    // Game info
    DrawText("GAME INFO", panel_x, panel_y, 18, RL_DARKBLUE);
    panel_y += 25;
    
    DrawText(TextFormat("Mode: %s", GAME_MODE_NAMES[mode]), panel_x, panel_y, 14, RL_BLACK);
    panel_y += 20;
    
    if (mode == GM_PLAYER_STOCKFISH || mode == GM_AGENT_STOCKFISH) {
        DrawText(TextFormat("Stockfish ELO: %d", elo_setting), panel_x, panel_y, 14, RL_BLACK);
        panel_y += 20;
    }
    
    // Current player
    const char* current_player = (ctx->board.side_to_move() == WHITE) ? "White" : "Black";
    DrawText(TextFormat("To move: %s", current_player), panel_x, panel_y, 14, RL_BLACK);
    panel_y += 20;
    
    // Game status
    if (env->terminals[0]) {
        DrawText("GAME OVER", panel_x, panel_y, 16, RL_RED);
        panel_y += 20;
    } else if (game_paused) {
        DrawText("PAUSED", panel_x, panel_y, 16, RL_ORANGE);
        panel_y += 20;
    }
    
    // Move count
    DrawText(TextFormat("Moves: %d", (int)game_moves.size()), panel_x, panel_y, 14, RL_BLACK);
    panel_y += 25;
    
    // Session statistics
    if (session_stats.total_games > 0) {
        DrawText("SESSION STATS", panel_x, panel_y, 16, RL_DARKBLUE);
        panel_y += 20;
        
        DrawText(TextFormat("Games: %d", session_stats.total_games), panel_x, panel_y, 14, RL_BLACK);
        panel_y += 18;
        
        DrawText(TextFormat("W/L/D: %d/%d/%d", session_stats.total_wins, session_stats.total_losses, session_stats.total_draws), panel_x, panel_y, 14, RL_BLACK);
        panel_y += 18;
    }
    
    // Controls
    panel_y += 10;
    DrawText("CONTROLS", panel_x, panel_y, 16, RL_DARKBLUE);
    panel_y += 20;
    
    DrawText("SPACE: Pause/Resume", panel_x, panel_y, 12, RL_DARKGRAY);
    panel_y += 16;
    DrawText("R: Reset game", panel_x, panel_y, 12, RL_DARKGRAY);
    panel_y += 16;
    DrawText("M: Return to menu", panel_x, panel_y, 12, RL_DARKGRAY);
    panel_y += 16;
    DrawText("S: Show statistics", panel_x, panel_y, 12, RL_DARKGRAY);
    panel_y += 16;
    DrawText("C: Clear statistics", panel_x, panel_y, 12, RL_DARKGRAY);
    panel_y += 16;
    DrawText("X: Save statistics", panel_x, panel_y, 12, RL_DARKGRAY);
}

int main() {
    using namespace chess;
    
    printf("PufferLib Chess Evaluation – GUI Menu Version\n");
    srand(static_cast<unsigned>(time(NULL)));

    // Initialize global game logger with training logs directory
    global_game_logger = new GameLogger("pufferlib/resources/chess/training_logs/complete_games");
    
    // Load agent weights once (used for all agent-controlled sides)
    const char *weights_path = "resources/chess/puffer_chess_weights.bin";
    Weights *weights_white = load_weights(weights_path, CHESS_NUM_WEIGHTS);
    Weights *weights_black = load_weights(weights_path, CHESS_NUM_WEIGHTS);
    if (!weights_white || !weights_black) {
        fprintf(stderr, "ERROR: Could not load weights at %s\n", weights_path);
        return 1;
    }
    ChessNet *agent_net_white = init_chessnet(weights_white, 1);
    ChessNet *agent_net_black = init_chessnet(weights_black, 1);

    // Setup Raylib window
    InitWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "PufferLib Chess – Menu");
    SetTargetFPS(60);

    // Load piece textures (shared between menu & game)
    ChessPieceTextures textures = load_piece_textures();

    // Game/environment objects (re-created when starting a match)
    CChess env = {0};
    global_env_ptr = &env; // Set global pointer to current environment
    ChessNet *white_net = nullptr;
    ChessNet *black_net = nullptr;
    int elo_setting = 1320; // default ELO for Stockfish (weak)

    bool in_menu       = true;
    int menu_index     = 0;
    GameMode game_mode = GM_PLAYER_STOCKFISH;

    static std::ofstream pgn_log("game_log.pgn", std::ios::app);

    while (!WindowShouldClose()) {
        BeginDrawing();
        if (in_menu) {
            // MAIN MENU RENDER + INPUT
            ClearBackground(RL_RAYWHITE);
            DrawText("PufferLib Chess", 50, 20, 32, RL_BLACK);
            DrawText("Use UP / DOWN to choose, LEFT / RIGHT to adjust, ENTER to start", 50, 60, 18, RL_DARKGRAY);
            DrawText("Player vs Random = Human (White) vs Random (Black)", 50, 85, 16, RL_DARKBLUE);

            for (int i = 0; i < GM_COUNT; ++i) {
                ::Color col = (i == menu_index) ? RL_RED : RL_BLACK;
                char menu_label[64] = {0};
                if (i == GM_PLAYER_STOCKFISH || i == GM_AGENT_STOCKFISH) {
                    snprintf(menu_label, sizeof(menu_label), "%s (ELO %d)", GAME_MODE_NAMES[i], elo_setting);
                } else {
                    snprintf(menu_label, sizeof(menu_label), "%s", GAME_MODE_NAMES[i]);
                }
                DrawText(menu_label, 80, 120 + i * 30, 20, col);
            }
            
            // Display session statistics in menu
            if (session_stats.total_games > 0) {
                int stats_y = 120 + GM_COUNT * 30 + 40;
                DrawText("=== Session Statistics ===", 50, stats_y, 18, RL_DARKBLUE);
                stats_y += 25;
                DrawText(TextFormat("Total Games: %d", session_stats.total_games), 50, stats_y, 16, RL_BLACK);
                stats_y += 20;
                DrawText(TextFormat("Overall W/L/D: %d/%d/%d", session_stats.total_wins, session_stats.total_losses, session_stats.total_draws), 50, stats_y, 16, RL_BLACK);
                stats_y += 25;
                
                // Show specific stats if available
                if (session_stats.agent_stats.games > 0) {
                    DrawText(TextFormat("Agent: %.1f%% win rate (%d/%d/%d)", 
                            session_stats.agent_stats.win_rate() * 100, session_stats.agent_stats.wins, 
                            session_stats.agent_stats.losses, session_stats.agent_stats.draws), 50, stats_y, 16, RL_DARKGREEN);
                    stats_y += 20;
                }
                if (session_stats.human_stats.games > 0) {
                    DrawText(TextFormat("Human: %.1f%% win rate (%d/%d/%d)", 
                            session_stats.human_stats.win_rate() * 100, session_stats.human_stats.wins,
                            session_stats.human_stats.losses, session_stats.human_stats.draws), 50, stats_y, 16, RL_DARKGREEN);
                    stats_y += 20;
                }
            }
            
            // Menu instructions
            DrawText("Press C to clear statistics, X to save to file", 50, WINDOW_HEIGHT - 50, 16, RL_DARKGRAY);

            // Input handling
            if (IsKeyPressed(KEY_UP))    menu_index = (menu_index + GM_COUNT - 1) % GM_COUNT;
            if (IsKeyPressed(KEY_DOWN))  menu_index = (menu_index + 1) % GM_COUNT;

            // Adjust ELO when Player vs Stockfish or Agent vs Stockfish is selected
            if (menu_index == GM_PLAYER_STOCKFISH || menu_index == GM_AGENT_STOCKFISH) {
                int delta = 0;
                if (IsKeyDown(KEY_LEFT))  delta -= 5;
                if (IsKeyDown(KEY_RIGHT)) delta += 5;
                if (delta != 0) {
                    elo_setting = std::clamp(elo_setting + delta, 300, 3500);
                }
            }

            // Menu keyboard shortcuts
            if (IsKeyPressed(KEY_C)) {
                session_stats.reset();
                session_wins = session_losses = session_draws = 0;
                printf("Session statistics cleared.\n");
            }
            
            if (IsKeyPressed(KEY_X)) {
                if (session_stats.total_games > 0) {
                    session_stats.save_to_file(static_cast<GameMode>(menu_index));
                } else {
                    printf("No statistics to save.\n");
                }
            }
            
            // Start game
            if (IsKeyPressed(KEY_ENTER)) {
                game_mode = static_cast<GameMode>(menu_index);
                
                game_moves.clear(); // Clear moves when starting a new game

                if (game_mode == GM_GAME_REPLAY) {
                    // ONLY Game Replay mode should access saved training games
                    printf("Entering Game Replay mode - loading saved training games...\n");
                    load_available_games();
                    
                    // Initialize environment for replay mode
                    env.reward_valid = 0.0f;
                    env.reward_invalid_white = -0.1f;
                    env.reward_invalid_black = -0.1f;
                    env.reward_agent_captures_enemy_piece = 0.05f;
                    env.reward_enemy_captures_agent_piece = -0.05f;
                    env.reward_draw = 0.0f;
                    env.reward_win_white = 1.0f;
                    env.reward_win_black = 1.0f;
                    env.reward_loss_white = -1.0f;
                    env.reward_loss_black = -1.0f;
                    env.reward_check_white = 0.0f;
                    env.reward_check_black = 0.0f;
                    env.reward_material_diff_white = 0.0f;
                    env.reward_material_diff_black = 0.0f;
                    env.max_depth = 200;
                    
                    allocate(&env);
                    init(&env);
                    c_reset(&env);
                    
                    // Disable Stockfish for replay mode
                    auto *ctx = (ChessContext *)env.context;
                    ctx->stockfish_enabled = false;
                    
                    in_menu = false;
                    show_game_list = true;
                    replay_mode_active = false;
                    selected_game_index = 0;
                    
                    printf("[REPLAY] Environment initialized for replay mode\n");
                } else {
                    // ALL other modes are live gameplay
                    printf("Starting %s mode...\n", GAME_MODE_NAMES[game_mode]);
                    in_menu = false;

                    // Initialize environment fresh for each match
                    env.reward_valid = 0.0f;
                    env.reward_invalid_white = -0.1f;
                    env.reward_invalid_black = -0.1f;
                    env.reward_agent_captures_enemy_piece = 0.05f;
                    env.reward_enemy_captures_agent_piece = -0.05f;
                    env.reward_draw = 0.0f;
                    env.reward_win_white = 1.0f;
                    env.reward_win_black = 1.0f;
                    env.reward_loss_white = -1.0f;
                    env.reward_loss_black = -1.0f;
                    env.reward_check_white = 0.0f;
                    env.reward_check_black = 0.0f;
                    env.reward_material_diff_white = 0.0f;
                    env.reward_material_diff_black = 0.0f;
                    env.max_depth = 200;

                    allocate(&env);
                    init(&env);
                    c_reset(&env);
                    
                    // For dual agent mode, we need to reallocate arrays for 2 agents
                    if (game_mode == GM_AGENT_AGENT) {
                        // Free single agent arrays
                        free(env.observations);
                        free(env.actions);
                        free(env.rewards);
                        free(env.terminals);
                        
                        // Allocate dual agent arrays
                        env.observations = (float*)calloc(2 * 6018, sizeof(float));
                        env.actions = (int*)calloc(2, sizeof(int));
                        env.rewards = (float*)calloc(2, sizeof(float));
                        env.terminals = (unsigned char*)calloc(2, sizeof(unsigned char));
                        
                        printf("Reallocated arrays for dual agent mode\n");
                    }

                    // Enable / disable Stockfish depending on mode
                    auto *ctx = (ChessContext *)env.context;
                    if (game_mode == GM_PLAYER_STOCKFISH || game_mode == GM_AGENT_STOCKFISH) {
                        // Initialize Stockfish engine
                        enable_stockfish_black(&env, nullptr, elo_setting, 10);
                        ctx->stockfish_enabled = true;
                        printf("Stockfish engine enabled (ELO %d)\n", elo_setting);
                    } else {
                        ctx->stockfish_enabled = false;
                        printf("Stockfish engine disabled\n");
                    }

                    // Set up AI agents based on game mode
                    white_net = (game_mode == GM_AGENT_STOCKFISH || game_mode == GM_AGENT_AGENT || game_mode == GM_AGENT_RANDOM) ? agent_net_white : nullptr;
                    black_net = (game_mode == GM_AGENT_AGENT || game_mode == GM_RANDOM_AGENT) ? agent_net_black : nullptr;
                    
                    // Configure environment mode based on game type
                    if (game_mode == GM_AGENT_AGENT) {
                        // Set dual agent mode for agent vs agent
                        ctx->dual_agent_self_play_mode = true;
                        ctx->self_play_mode = false;
                        printf("Configured for dual agent self-play mode\n");
                    } else {
                        // Use single agent mode for all other modes
                        ctx->dual_agent_self_play_mode = false;
                        ctx->self_play_mode = false;
                        printf("Configured for single agent mode\n");
                    }
                }
            }
        } else if (show_game_list) {
            // Game list selection screen
            render_game_list_screen();
            handle_game_list_input();
            
            if (!show_game_list && !replay_mode_active) {
                in_menu = true;
            }
        } else if (replay_mode_active) {
            // Game replay screen
            render_game_replay_screen(&env, &textures);
            handle_game_replay_input();
            
            if (!replay_mode_active) {
                show_game_list = true;
            }
        } else {
            // GAMEPLAY RENDER + LOGIC
            auto *ctx = (ChessContext *)env.context;
            
            // Toggle pause
            if (IsKeyPressed(KEY_SPACE)) {
                game_paused = !game_paused;
                printf("Game %s\n", game_paused ? "PAUSED" : "RESUMED");
            }
            
            // Toggle best move display
            if (IsKeyPressed(KEY_B)) show_bestmove = !show_bestmove;
            
            int sim_steps = IsKeyDown(KEY_RIGHT) ? 8 : 1; // speed-up while holding RIGHT
            
            // Only simulate if not paused
            if (!game_paused) {
                for (int step = 0; step < sim_steps && !env.terminals[0]; ++step) {

                    // DUAL AGENT MODE: Handle both agents simultaneously
                    if (game_mode == GM_AGENT_AGENT) {
                        // Compute dual agent observations first
                        compute_dual_agent_observations(&env, ctx);
                        
                        // Get actions from both agents using correct observation offsets
                        int white_action = agent_select_action_dual(white_net, &env, 0);
                        int black_action = agent_select_action_dual(black_net, &env, 1);
                        
                        env.actions[0] = white_action;
                        env.actions[1] = black_action;
                        
                        // Record move for current player
                        chess::Color current_player = ctx->board.side_to_move();
                        int current_action = (current_player == WHITE) ? white_action : black_action;
                        Move mv = action_to_move_lookup(current_action, ctx->board);
                        if (mv.from.x >= 0) {
                            game_moves.push_back(move_to_uci(mv));
                            printf("[DUAL_AGENT] Move: %s (action %d) by %s\n", 
                                   move_to_uci(mv).c_str(), current_action, 
                                   (current_player == WHITE) ? "White" : "Black");
                        }
                        
                        c_step(&env);
                        continue;
                    }

                    // SINGLE AGENT MODES: Handle one side at a time
                    if (ctx->board.side_to_move() == WHITE) {
                        switch (game_mode) {
                            case GM_PLAYER_STOCKFISH:
                            case GM_PLAYER_RANDOM: {
                                // Human move handled via mouse clicks - no automatic action here
                                break; 
                            }
                            case GM_AGENT_STOCKFISH:
                            case GM_AGENT_RANDOM: {
                                // Compute observations for agent
                                compute_observation(&env, ctx);
                                
                                int chosen_action;
                                if (white_net) chosen_action = agent_select_action(white_net, &env);
                                else chosen_action = random_select_action(ctx);
                                env.actions[0] = chosen_action;

                                // Record move
                                Move mv = action_to_move_lookup(chosen_action, ctx->board);
                                if (mv.from.x >= 0) {
                                    game_moves.push_back(move_to_uci(mv));
                                    printf("[AGENT_WHITE] Move: %s (action %d)\n", move_to_uci(mv).c_str(), chosen_action);
                                }

                                c_step(&env);
                                break; 
                            }
                            case GM_RANDOM_RANDOM: {
                                env.actions[0] = random_select_action(ctx);
                                {
                                    auto mv = action_to_move_lookup(env.actions[0], ctx->board);
                                    if (mv.from.x >= 0) game_moves.push_back(move_to_uci(mv));
                                }
                                c_step(&env);
                                break; 
                            }
                            case GM_RANDOM_AGENT: {
                                env.actions[0] = random_select_action(ctx);
                                {
                                    auto mv = action_to_move_lookup(env.actions[0], ctx->board);
                                    if (mv.from.x >= 0) game_moves.push_back(move_to_uci(mv));
                                }
                                c_step(&env);
                                break; 
                            }
                            default: break;
                        }
                    } else { // Black to move
                        switch (game_mode) {
                            case GM_PLAYER_STOCKFISH: {
                                // Always call c_step for Stockfish to make its move
                                c_step(&env);
                                break; 
                            }
                            case GM_PLAYER_RANDOM: {
                                // Random black move
                                env.actions[0] = random_select_action(ctx);
                                {
                                    auto mv = action_to_move_lookup(env.actions[0], ctx->board);
                                    if (mv.from.x >= 0) {
                                        game_moves.push_back(move_to_uci(mv));
                                        printf("[PLAYER_RANDOM] Random opponent move: %c%d -> %c%d (action %d)\n", 
                                               'a' + mv.from.x, mv.from.y + 1, 'a' + mv.to.x, mv.to.y + 1, env.actions[0]);
                                    }
                                }
                                c_step(&env);
                                break; 
                            }
                            case GM_AGENT_STOCKFISH: {
                                // Always call c_step for Stockfish to make its move
                                c_step(&env);
                                break; 
                            }
                            case GM_AGENT_RANDOM: {
                                env.actions[0] = random_select_action(ctx);
                                {
                                    auto mv = action_to_move_lookup(env.actions[0], ctx->board);
                                    if (mv.from.x >= 0) game_moves.push_back(move_to_uci(mv));
                                }
                                c_step(&env);
                                break; 
                            }
                            case GM_RANDOM_RANDOM: {
                                env.actions[0] = random_select_action(ctx);
                                {
                                    auto mv = action_to_move_lookup(env.actions[0], ctx->board);
                                    if (mv.from.x >= 0) game_moves.push_back(move_to_uci(mv));
                                }
                                c_step(&env);
                                break; 
                            }
                            case GM_RANDOM_AGENT: {
                                // Agent plays black in this mode
                                compute_observation(&env, ctx);
                                
                                int chosen_action;
                                if (black_net) chosen_action = agent_select_action(black_net, &env);
                                else chosen_action = random_select_action(ctx);
                                env.actions[0] = chosen_action;
                                {
                                    auto mv = action_to_move_lookup(chosen_action, ctx->board);
                                    if (mv.from.x >= 0) {
                                        game_moves.push_back(move_to_uci(mv));
                                        printf("[AGENT_BLACK] Move: %s (action %d)\n", move_to_uci(mv).c_str(), chosen_action);
                                    }
                                }
                                c_step(&env);
                                break; 
                            }
                            default: break;
                        }
                    }

                    // Check if a game just ended by looking at the captured outcome
                    if (chess::last_game_outcome.game_ended && env.terminals[0]) {
                        printf("[DEBUG] Game outcome captured! white_won=%d black_won=%d is_draw=%d reason='%s'\n", 
                               chess::last_game_outcome.white_won, chess::last_game_outcome.black_won, 
                               chess::last_game_outcome.is_draw, chess::last_game_outcome.draw_reason.c_str());
                        
                        // Update session statistics
                        update_session_stats(game_mode, chess::last_game_outcome.white_won, chess::last_game_outcome.black_won, chess::last_game_outcome.is_draw);
                        
                        // Clear the outcome for next game
                        chess::last_game_outcome = GameOutcome();
                        
                        // Write PGN-like line (simple UCI list) to log
                        if (pgn_log.is_open()) {
                            std::ostringstream oss;
                            int move_num = 1;
                            for (size_t i = 0; i < game_moves.size(); ++i) {
                                if (i % 2 == 0) oss << move_num++ << ".";
                                oss << game_moves[i] << " ";
                            }
                            oss << "\n";
                            pgn_log << oss.str();
                            pgn_log.flush();
                        }

                        // Start new game automatically in same mode
                        game_moves.clear();
                        c_reset(&env);
                        printf("[GAME_RESET] Starting new game in %s mode\n", GAME_MODE_NAMES[game_mode]);
                    }
                }
            } // End of if (!game_paused)

            // Handle player input for human moves (only if it's player's turn and correct mode)
            if ((game_mode == GM_PLAYER_STOCKFISH || game_mode == GM_PLAYER_RANDOM) && ctx->board.side_to_move() == WHITE) {
                if (IsMouseButtonPressed(MOUSE_LEFT_BUTTON)) {
                    Vector2 mp = GetMousePosition();
                    int bx = (mp.x - 50) / 64;
                    int by = (mp.y - 70) / 64;
                    static int sel_fx = -1, sel_fy = -1;
                    static bool selecting = false;

                    if (bx >= 0 && bx < 8 && by >= 0 && by < 8) {
                        int board_x = bx;
                        int board_y = 7 - by;

                        if (!selecting) {
                            Square pos{(int8_t)board_x, (int8_t)board_y};
                            const Piece &p = ctx->board.at(pos);
                            if (p.color == WHITE && p.type != EMPTY) {
                                sel_fx = board_x;
                                sel_fy = board_y;
                                selecting = true;
                            }
                        } else {
                            const auto &legal = ctx->board.legal_moves();
                            Move chosen = kPassMove;
                            for (const auto &mv : legal) {
                                if (mv.from.x == sel_fx && mv.from.y == sel_fy && mv.to.x == board_x && mv.to.y == board_y) {
                                    chosen = mv; break; }
                            }
                            selecting = false;
                            if (!(chosen == kPassMove)) {
                                // Check for promotion move
                                if (is_promotion_move(ctx->board, sel_fx, sel_fy, board_x, board_y)) {
                                    show_promotion_selection = true;
                                    promotion_from_x = sel_fx;
                                    promotion_from_y = sel_fy;
                                    promotion_to_x = board_x;
                                    promotion_to_y = board_y;
                                    selected_promotion = QUEEN; // Default selection
                                } else {
                                    env.actions[0] = ChessBoard::move_to_action(chosen);
                                    
                                    // Record move before applying
                                    game_moves.push_back(move_to_uci(chosen));
                                    
                                    c_step(&env);
                                }
                            }
                        }
                    }
                }
            }

            // Allow reset / return to menu
            if (IsKeyPressed(KEY_R)) {
                game_moves.clear();
                c_reset(&env);
                printf("[MANUAL_RESET] Game reset by user\n");
            }
            
            // Print detailed statistics summary
            if (IsKeyPressed(KEY_S)) {
                session_stats.print_summary(game_mode);
            }
            
            // Reset session statistics
            if (IsKeyPressed(KEY_C)) {
                session_stats.reset();
                session_wins = session_losses = session_draws = 0;
                printf("Session statistics cleared.\n");
            }
            
            // Save statistics to file
            if (IsKeyPressed(KEY_X)) {
                session_stats.save_to_file(game_mode);
            }

            if (IsKeyPressed(KEY_M)) {
                // Clean up current game resources and safely return to menu
                c_close(&env);
                free_allocated(&env);
                memset(&env, 0, sizeof(env));

                game_moves.clear(); // Clear moves when returning to menu
                in_menu = true;
                EndDrawing();
                continue;
            }

            // Render board / UI
            ClearBackground(RL_RAYWHITE);
            render_chess_board(&env, &textures);
            draw_side_panel(&env, &textures, game_mode, elo_setting, white_net, black_net);
            DrawText(TextFormat("M:Menu  R:Reset  S:Stats  C:Clear  X:Save  →:Speed  SPACE:Pause"), 10, WINDOW_HEIGHT - 30, 16, RL_DARKGRAY);
            
            // Render promotion selection dialog if active
            if (show_promotion_selection) {
                render_promotion_selection(&textures);
                handle_promotion_selection();
            }
        }
        EndDrawing();
    }

    // Auto-save statistics before exit if any games were played
    if (session_stats.total_games > 0) {
        printf("Auto-saving session statistics...\n");
        session_stats.save_to_file(game_mode, "chess_eval_stats_autosave.txt");
        session_stats.print_summary(game_mode);
    }

    // Cleanup game logger
    if (global_game_logger) {
        delete global_game_logger;
        global_game_logger = nullptr;
    }

    // Cleanup resources
    unload_piece_textures(&textures);
    free_chessnet(agent_net_white);
    free_chessnet(agent_net_black);
    CloseWindow();

    return 0;
}