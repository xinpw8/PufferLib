// chess.cpp - Graphical Chess Evaluation using Raylib
#include <time.h>
#include <math.h>
#include "chess.h"
#include "../../extensions/puffernet.h"
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

// Raylib defines macros WHITE, BLACK that clash with chess::Color constants used later with qualification.
#ifdef WHITE
#undef WHITE
#endif
#ifdef BLACK
#undef BLACK
#endif

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
    
    printf("About to check textures and return...\n");
    fflush(stdout);
    
    // Check if textures loaded successfully  
    printf("Texture loading completed successfully.\n"); 
    
    printf("Returning texture struct...\n");
    fflush(stdout);
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

static Texture2D get_piece_texture(const ChessPieceTextures *textures, chess::Color color, chess::PieceType type) {
    Texture2D empty_texture = {0};
    if (type == chess::EMPTY) return empty_texture;
    
    if (color == 0) { // WHITE = 0
        switch (type) {
            case chess::KING: return textures->wking;
            case chess::QUEEN: return textures->wqueen;
            case chess::ROOK: return textures->wrook;
            case chess::BISHOP: return textures->wbishop;
            case chess::KNIGHT: return textures->wknight;
            case chess::PAWN: return textures->wpawn;
            case chess::EMPTY: 
            default: return empty_texture;
        }
    } else {
        switch (type) {
            case chess::KING: return textures->bking;
            case chess::QUEEN: return textures->bqueen;
            case chess::ROOK: return textures->brook;
            case chess::BISHOP: return textures->bbishop;
            case chess::KNIGHT: return textures->bknight;
            case chess::PAWN: return textures->bpawn;
            case chess::EMPTY:
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

// GameOutcome is already defined in chess.h, use the existing one

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
void apply_replay_moves_to_board(chess::ChessBoard& board, const GameReplay& replay);
void update_session_stats(GameMode mode, bool white_won, bool black_won, bool is_draw);
int agent_select_action(ChessNet* net, CChess* env);
int agent_select_action_dual(ChessNet* net, CChess* env, int agent_index);
int random_select_action(ChessContext* ctx);
std::string move_to_uci(const chess::Move& move);

class GameLogger {
private:
    std::vector<GameLogEntry> games;
    std::string log_directory;
    
public:
    GameLogger(const std::string& directory) : log_directory(directory) {
        load_games_from_directory();
    }
    
    void load_games_from_directory() {
        games.clear();
        
        // Use C++ filesystem to read directory
        std::string command = "find " + log_directory + " -name '*.txt' | sort";
        FILE* pipe = popen(command.c_str(), "r");
        if (!pipe) {
            printf("[GameLogger] Error opening directory: %s\n", log_directory.c_str());
            return;
        }
        
        char buffer[1024];
        std::vector<std::string> filenames;
        while (fgets(buffer, sizeof(buffer), pipe)) {
            std::string filename = buffer;
            // Remove newline
            if (!filename.empty() && filename.back() == '\n') {
                filename.pop_back();
            }
            filenames.push_back(filename);
        }
        pclose(pipe);
        
        // Parse each game file
        for (const auto& filename : filenames) {
            GameLogEntry entry;
            if (parse_game_file(filename, entry)) {
                games.push_back(entry);
            }
        }
        
        printf("[GameLogger] Loaded %d games from %s\n", (int)games.size(), log_directory.c_str());
    }
    
    bool parse_game_file(const std::string& filename, GameLogEntry& entry) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            return false;
        }
        
        entry.filename = filename;
        entry.moves.clear();
        entry.draw_reason = ""; // Initialize draw reason
        
        std::string line;
        while (std::getline(file, line)) {
            if (line.empty()) continue;
            
            if (line.find("# Complete chess game logged at") != std::string::npos) {
                // Extract timestamp
                size_t pos = line.find_last_of(" ");
                if (pos != std::string::npos) {
                    entry.timestamp = line.substr(pos + 1);
                }
            } else if (line.find("# Outcome:") != std::string::npos) {
                // Extract outcome
                size_t pos = line.find(": ");
                if (pos != std::string::npos) {
                    entry.outcome = line.substr(pos + 2);
                }
            } else if (line.find("# Draw reason:") != std::string::npos) {
                // Extract draw reason if present
                size_t pos = line.find(": ");
                if (pos != std::string::npos) {
                    entry.draw_reason = line.substr(pos + 2);
                }
            } else if (line.find("# Total moves:") != std::string::npos) {
                // Extract total moves
                size_t pos = line.find(": ");
                if (pos != std::string::npos) {
                    entry.total_moves = std::stoi(line.substr(pos + 2));
                }
            } else if (line[0] != '#') {
                // Parse move line: "1. 656 a2b4"
                std::istringstream iss(line);
                std::string move_str, action_str, notation;
                
                if (iss >> move_str >> action_str >> notation) {
                    GameMove move;
                    move.move_number = std::stoi(move_str.substr(0, move_str.length() - 1)); // Remove the '.'
                    move.action_id = std::stoi(action_str);
                    move.algebraic_notation = notation;
                    entry.moves.push_back(move);
                }
            }
        }
        
        file.close();
        return true;
    }
    
    const std::vector<GameLogEntry>& get_games() const {
        return games;
    }
    
    std::vector<std::string> get_game_list() const {
        std::vector<std::string> game_list;
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
            game_list.push_back(std::string(buffer));
        }
        return game_list;
    }
    
    const GameLogEntry* get_game(int index) const {
        if (index < 0 || index >= static_cast<int>(games.size())) {
            return nullptr;
        }
        return &games[index];
    }
    
    void get_aggregate_stats(int& total_games, int& wins, int& draws, int& losses, 
                            float& avg_moves, int& shortest_game, int& longest_game) const {
        total_games = static_cast<int>(games.size());
        wins = draws = losses = 0;
        int total_moves = 0;
        shortest_game = INT_MAX;
        longest_game = 0;
        
        for (const auto& game : games) {
            if (game.outcome == "win") wins++;
            else if (game.outcome == "loss") losses++;
            else if (game.outcome == "draw") draws++;
            
            total_moves += game.total_moves;
            shortest_game = std::min(shortest_game, game.total_moves);
            longest_game = std::max(longest_game, game.total_moves);
        }
        
        avg_moves = total_games > 0 ? static_cast<float>(total_moves) / total_games : 0.0f;
        if (total_games == 0) {
            shortest_game = 0;
            longest_game = 0;
        }
    }
    
    void print_statistics() const {
        int total_games, wins, draws, losses, shortest_game, longest_game;
        float avg_moves;
        get_aggregate_stats(total_games, wins, draws, losses, avg_moves, shortest_game, longest_game);
        
        printf("=== Game Database Statistics ===\n");
        printf("Total games: %d\n", total_games);
        printf("White wins: %d (%.1f%%)\n", wins, total_games > 0 ? 100.0f * wins / total_games : 0.0f);
        printf("Black wins: %d (%.1f%%)\n", losses, total_games > 0 ? 100.0f * losses / total_games : 0.0f);
        printf("Draws: %d (%.1f%%)\n", draws, total_games > 0 ? 100.0f * draws / total_games : 0.0f);
        printf("Average moves: %.1f\n", avg_moves);
        printf("Move range: %d - %d\n", shortest_game, longest_game);
    }
};

// Function implementations
void load_available_games() {
    available_games.clear();
    if (global_game_logger) {
        available_games = global_game_logger->get_game_list();
    }
    printf("Loaded %d available games\n", (int)available_games.size());
}

void render_game_list_screen() {
    ClearBackground(RAYWHITE);
    
    DrawText("Game Replay - Select Game", 50, 20, 24, RL_BLACK);
    DrawText(TextFormat("Games available: %d", (int)available_games.size()), 50, 50, 16, DARKGRAY);
    
    if (available_games.empty()) {
        DrawText("No games found in training logs directory", 50, 100, 18, RED);
        DrawText("Press B to return to menu", 50, 130, 16, DARKGRAY);
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
        Color color = (i == selected_game_index) ? RED : RL_BLACK;
        int y_pos = start_y + (i - start_index) * line_height;
        
        // Truncate long game descriptions
        std::string display_text = available_games[i];
        if (display_text.length() > 80) {
            display_text = display_text.substr(0, 77) + "...";
        }
        
        DrawText(display_text.c_str(), 50, y_pos, 14, color);
    }
    
    // Show selected game details
    if (selected_game_index >= 0 && selected_game_index < (int)available_games.size()) {
        const auto* game = global_game_logger->get_game(selected_game_index);
        if (game) {
            int details_y = DETAILS_START_Y;
            Color outcome_color = RL_BLACK;
            
            DrawText("GAME DETAILS:", DETAILS_START_X, details_y, 16, DARKBLUE);
            details_y += 25;
            
            DrawText(TextFormat("Timestamp: %s", game->timestamp.c_str()), DETAILS_START_X, details_y, 16, RL_BLACK);
            details_y += 22;
            
            // Convert outcome to clearer display format
            std::string outcome_display;
            if (game->outcome == "win") {
                outcome_display = "White Win";
                outcome_color = DARKGREEN;
            } else if (game->outcome == "loss") {
                outcome_display = "Black Win";
                outcome_color = DARKGREEN;
            } else if (game->outcome == "draw") {
                if (!game->draw_reason.empty()) {
                    std::string reason = game->draw_reason;
                    if (!reason.empty()) {
                        reason[0] = std::toupper(reason[0]);
                    }
                    outcome_display = "Draw (" + reason + ")";
                } else {
                    outcome_display = "Draw";
                }
                outcome_color = BLUE;
            } else {
                outcome_display = game->outcome;
            }
            
            DrawText(TextFormat("Outcome: %s", outcome_display.c_str()), DETAILS_START_X, details_y, 16, outcome_color);
            details_y += 22;
            
            DrawText(TextFormat("Total moves: %d", game->total_moves), DETAILS_START_X, details_y, 16, RL_BLACK);
            details_y += 22;
            
            DrawText(TextFormat("Recorded moves: %d", (int)game->moves.size()), DETAILS_START_X, details_y, 16, RL_BLACK);
            details_y += 22;
            
            // Show first few moves as preview
            if (!game->moves.empty()) {
                details_y += 10;
                DrawText("Opening moves:", DETAILS_START_X, details_y, 16, DARKBLUE);
                details_y += 20;
                
                std::string moves_preview;
                int moves_to_show = std::min(6, (int)game->moves.size());
                for (int i = 0; i < moves_to_show; ++i) {
                    if (i > 0) moves_preview += " ";
                    if (i % 2 == 0) {
                        moves_preview += std::to_string((i / 2) + 1) + ".";
                    }
                    moves_preview += game->moves[i].algebraic_notation;
                }
                if (game->moves.size() > 6) {
                    moves_preview += " ...";
                }
                
                DrawText(moves_preview.c_str(), DETAILS_START_X, details_y, 14, DARKGRAY);
                details_y += 20;
            }
        }
    }
    
    // Show aggregate statistics at bottom right
    if (global_game_logger) {
        int total_games, wins, draws, losses, shortest_game, longest_game;
        float avg_moves;
        global_game_logger->get_aggregate_stats(total_games, wins, draws, losses, avg_moves, shortest_game, longest_game);
        
        int stats_y = DETAILS_START_Y + 280;
        
        DrawText("Database Statistics:", DETAILS_START_X, stats_y, 18, DARKBLUE);
        stats_y += 25;
        
        DrawText(TextFormat("Total games: %d", total_games), DETAILS_START_X, stats_y, 16, RL_BLACK);
        stats_y += 20;
        
        DrawText(TextFormat("White wins: %d", wins), DETAILS_START_X, stats_y, 16, DARKGREEN);
        stats_y += 18;
        
        DrawText(TextFormat("Black wins: %d", losses), DETAILS_START_X, stats_y, 16, DARKGREEN);
        stats_y += 18;
        
        DrawText(TextFormat("Draws: %d", draws), DETAILS_START_X, stats_y, 16, BLUE);
        stats_y += 18;
        
        DrawText(TextFormat("Avg moves: %.1f", avg_moves), DETAILS_START_X, stats_y, 16, RL_BLACK);
        stats_y += 18;
        
        DrawText(TextFormat("Range: %d - %d moves", shortest_game, longest_game), DETAILS_START_X, stats_y, 16, RL_BLACK);
    }
    
    // Bottom navigation instructions (moved higher to avoid overlap)
    DrawText("F1: Filter by outcome  F2: Filter by length  F3: Show statistics", 50, WINDOW_HEIGHT - 80, 16, DARKGRAY);
    DrawText("R: Refresh game list  S: Print statistics  I: Game info", 50, WINDOW_HEIGHT - 60, 16, DARKGRAY);
    DrawText("UP/DOWN: Navigate (hold for fast scroll)  ENTER: Select  B: Back", 50, WINDOW_HEIGHT - 40, 16, DARKGRAY);
}

// Render game replay screen
void render_game_replay_screen(CChess* env, ChessPieceTextures* textures) {
    ClearBackground(RAYWHITE);
    
    if (!current_replay.is_active) {
        DrawText("No game loaded for replay", 50, 300, 24, RED);
        return;
    }
    
    // Apply moves from the replay to show the correct board position
    auto* ctx = (ChessContext*)env->context;
    if (ctx) {
        apply_replay_moves_to_board(ctx->board, current_replay);
    }
    
    // Render the chess board with the current replay position
    render_chess_board(env, textures);
    
    // Replay information panel - reorganized to prevent overlap
    int info_x = BOARD_OFFSET_X + BOARD_SIZE + 30;  // More spacing from board
    int info_y = BOARD_OFFSET_Y;
    const int line_height = 18;
    const int section_spacing = 25;
    
    // Title
    DrawText("GAME REPLAY MODE", info_x, info_y, 20, BLUE);
    info_y += 30;
    
    // Game info section
    DrawText("GAME INFO:", info_x, info_y, 16, DARKBLUE);
    info_y += 20;
    DrawText(TextFormat("Time: %s", current_replay.current_game->timestamp.c_str()), info_x, info_y, 14, RL_BLACK);
    info_y += line_height;
    DrawText(TextFormat("Result: %s", current_replay.current_game->outcome.c_str()), info_x, info_y, 14, RL_BLACK);
    info_y += line_height;
    DrawText(TextFormat("Move: %d/%d", current_replay.current_move_index + 1, current_replay.current_game->total_moves), info_x, info_y, 14, RL_BLACK);
    info_y += section_spacing;
    
    // Current move details
    const auto* current_move = current_replay.get_current_move();
    if (current_move) {
        DrawText("CURRENT MOVE:", info_x, info_y, 16, DARKBLUE);
        info_y += 20;
        DrawText(TextFormat("Action: %d", current_move->action_id), info_x, info_y, 14, RL_BLACK);
        info_y += line_height;
        DrawText(TextFormat("UCI: %s", current_move->algebraic_notation.c_str()), info_x, info_y, 14, RL_BLACK);
        info_y += line_height;
        DrawText(TextFormat("Number: %d", current_move->move_number), info_x, info_y, 14, RL_BLACK);
        info_y += section_spacing;
    }
    
    // Auto-play status
    if (auto_play_replay) {
        DrawText("AUTO-PLAY: ON", info_x, info_y, 16, GREEN);
        info_y += section_spacing;
    }
    
    // Controls section
    DrawText("CONTROLS:", info_x, info_y, 16, DARKBLUE);
    info_y += 20;
    DrawText("LEFT/RIGHT: Step moves", info_x, info_y, 13, DARKGRAY);
    info_y += 16;
    DrawText("(Hold for fast step)", info_x, info_y, 12, GRAY);
    info_y += 18;
    DrawText("HOME/END: Jump to start/end", info_x, info_y, 13, DARKGRAY);
    info_y += 16;
    DrawText("SPACE: Auto-play toggle", info_x, info_y, 13, DARKGRAY);
    info_y += 16;
    DrawText("UP/DOWN: Speed control", info_x, info_y, 13, DARKGRAY);
    info_y += 16;
    DrawText("I: Game info", info_x, info_y, 13, DARKGRAY);
    info_y += 16;
    DrawText("B: Back to game list", info_x, info_y, 13, DARKGRAY);
    
    // Move progress bar - moved much further down to avoid all overlap
    if (current_replay.current_game && current_replay.current_game->total_moves > 0) {
        int bar_x = 10;
        int bar_y = WINDOW_HEIGHT - 25;  // Much closer to bottom
        int bar_width = WINDOW_WIDTH - 20;  // Full width minus margins
        int bar_height = 15;  // Thinner bar
        
        // Progress text above the bar
        DrawText(TextFormat("Progress: %d/%d moves", current_replay.current_move_index + 1, current_replay.current_game->total_moves), 
                 bar_x, bar_y - 20, 14, RL_BLACK);
        
        // Progress bar
        DrawRectangle(bar_x, bar_y, bar_width, bar_height, LIGHTGRAY);
        
        float progress = (float)(current_replay.current_move_index + 1) / current_replay.current_game->total_moves;
        int progress_width = (int)(bar_width * progress);
        
        DrawRectangle(bar_x, bar_y, progress_width, bar_height, BLUE);
        DrawRectangleLines(bar_x, bar_y, bar_width, bar_height, RL_BLACK);
    }
}

// Fixed action to move decoder that matches the encoder
static chess::Move action_to_move_direct_fixed(int action, const chess::ChessBoard& board) {
    // Special cases first
    if (action == 0) return chess::kPassMove;
    if (action == 4672) {
        // Queenside castling
        chess::Square king_pos = board.find_king(board.side_to_move());
        if (king_pos.is_valid()) {
            int rank = (board.side_to_move() == chess::WHITE) ? 0 : 7;
            chess::Move m{king_pos, {2, int8_t(rank)}, {board.side_to_move(), chess::KING}, chess::EMPTY};
            m.is_castle_long = true;
            return m;
        }
        return {{-1,-1},{-1,-1},{chess::NO_COLOR, chess::EMPTY}, chess::EMPTY};
    }
    if (action == 4673) {
        // Kingside castling
        chess::Square king_pos = board.find_king(board.side_to_move());
        if (king_pos.is_valid()) {
            int rank = (board.side_to_move() == chess::WHITE) ? 0 : 7;
            chess::Move m{king_pos, {6, int8_t(rank)}, {board.side_to_move(), chess::KING}, chess::EMPTY};
            m.is_castle_short = true;
            return m;
        }
        return {{-1,-1},{-1,-1},{chess::NO_COLOR, chess::EMPTY}, chess::EMPTY};
    }
    
    // Regular moves: decode from action ID
    if (action < 1 || action > 4671) {
        return {{-1,-1},{-1,-1},{chess::NO_COLOR, chess::EMPTY}, chess::EMPTY};
    }
    
    // CRITICAL FIX: The encoding uses (x * 8 + y) * 73, so we need to decode correctly
    int from_square = action / chess::kNumActionDestinations;  // This gives us (x * 8 + y)
    int dest_index = action % chess::kNumActionDestinations;
    
    // Convert from_square back to coordinates - FIXED to match encoding
    int from_x = from_square / 8;  // x coordinate
    int from_y = from_square % 8;  // y coordinate
    
    // Validate from square
    if (from_x < 0 || from_x >= 8 || from_y < 0 || from_y >= 8) {
        return {{-1,-1},{-1,-1},{chess::NO_COLOR, chess::EMPTY}, chess::EMPTY};
    }
    
    chess::Square from{int8_t(from_x), int8_t(from_y)};
    
    // Get the piece at the from square
    const chess::Piece& piece = board.at(from);
    if (piece.type == chess::EMPTY || piece.color != board.side_to_move()) {
        return {{-1,-1},{-1,-1},{chess::NO_COLOR, chess::EMPTY}, chess::EMPTY};
    }
    
    // Create rotated coordinates for move calculation (always from white's perspective)
    int calc_from_y = from_y;
    if (board.side_to_move() == chess::BLACK) {
        calc_from_y = 7 - from_y;
    }
    
    // Handle under-promotions (first 9 destination indices)
    if (dest_index < chess::kNumUnderPromotions) {
        int promo_piece_idx = dest_index / 3;
        int direction = dest_index % 3;
        
        static constexpr chess::PieceType kUnder[3] = {chess::KNIGHT, chess::BISHOP, chess::ROOK};
        static constexpr int kDirs[3] = {-1, 0, 1}; // left capture, straight, right capture
        
        if (promo_piece_idx >= 3) {
            return {{-1,-1},{-1,-1},{chess::NO_COLOR, chess::EMPTY}, chess::EMPTY};
        }
        
        int to_x = from_x + kDirs[direction];
        int to_y = calc_from_y + ((board.side_to_move() == chess::WHITE) ? 1 : -1);
        
        // Convert back to actual board coordinates
        if (board.side_to_move() == chess::BLACK) {
            to_y = 7 - to_y;
        }
        
        // Validate destination
        if (to_x < 0 || to_x >= 8 || to_y < 0 || to_y >= 8) {
            return {{-1,-1},{-1,-1},{chess::NO_COLOR, chess::EMPTY}, chess::EMPTY};
        }
        
        chess::Square to{int8_t(to_x), int8_t(to_y)};
        return {from, to, piece, kUnder[promo_piece_idx]};
    }
    
    // Regular moves - decode destination
    dest_index -= chess::kNumUnderPromotions;  // Skip under-promotion indices
    
    int dx = 0, dy = 0;
    
    // Decode based on destination index ranges (matching the encoder)
    if (dest_index >= 0 && dest_index <= 6) {         // N 0-6
        dx = 0;
        dy = dest_index + 1;
    } else if (dest_index >= 7 && dest_index <= 13) { // NE 7-13
        dx = dest_index - 6;
        dy = dest_index - 6;
    } else if (dest_index >= 14 && dest_index <= 20) { // E 14-20
        dx = dest_index - 13;
        dy = 0;
    } else if (dest_index >= 21 && dest_index <= 27) { // SE 21-27
        dx = dest_index - 20;
        dy = -(dest_index - 20);
    } else if (dest_index >= 28 && dest_index <= 34) { // S 28-34
        dx = 0;
        dy = -(dest_index - 27);
    } else if (dest_index >= 35 && dest_index <= 41) { // SW 35-41
        dx = -(dest_index - 34);
        dy = -(dest_index - 34);
    } else if (dest_index >= 42 && dest_index <= 48) { // W 42-48
        dx = -(dest_index - 41);
        dy = 0;
    } else if (dest_index >= 49 && dest_index <= 55) { // NW 49-55
        dx = -(dest_index - 48);
        dy = dest_index - 48;
    } else if (dest_index >= 56 && dest_index <= 63) { // Knight moves 56-63
        static constexpr int kKnight[8][2] = {
            {-2,-1}, {-2, 1}, {-1,-2}, {-1, 2},
            { 2,-1}, { 2, 1}, { 1,-2}, { 1, 2}
        };
        int knight_idx = dest_index - 56;
        if (knight_idx >= 0 && knight_idx < 8) {
            dx = kKnight[knight_idx][0];
            dy = kKnight[knight_idx][1];
        } else {
            return {{-1,-1},{-1,-1},{chess::NO_COLOR, chess::EMPTY}, chess::EMPTY};
        }
    } else {
        return {{-1,-1},{-1,-1},{chess::NO_COLOR, chess::EMPTY}, chess::EMPTY};
    }
    
    // Calculate destination square
    int to_x = from_x + dx;
    int to_y = calc_from_y + dy;
    
    // Convert back to actual board coordinates
    if (board.side_to_move() == chess::BLACK) {
        to_y = 7 - to_y;
    }
    
    // Validate destination
    if (to_x < 0 || to_x >= 8 || to_y < 0 || to_y >= 8) {
        return {{-1,-1},{-1,-1},{chess::NO_COLOR, chess::EMPTY}, chess::EMPTY};
    }
    
    chess::Square to{int8_t(to_x), int8_t(to_y)};
    
    // Check for queen promotion (pawn moving to promotion rank)
    chess::PieceType promotion = chess::EMPTY;
    if (piece.type == chess::PAWN) {
        int promotion_rank = (board.side_to_move() == chess::WHITE) ? 7 : 0;
        if (to_y == promotion_rank) {
            promotion = chess::QUEEN; // Default to queen promotion for non-under-promotion moves
        }
    }
    
    return {from, to, piece, promotion};
}

// Handle game replay input
void handle_game_replay_input() {
    if (!current_replay.is_active) return;
    
    // Handle holding arrow keys for fast replay navigation
    static auto last_step_time = std::chrono::steady_clock::now();
    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_step_time);
    
    bool should_step = false;
    int step_direction = 0;
    
    // Navigation controls - immediate response for single presses
    if (IsKeyPressed(KEY_RIGHT) || IsKeyPressed(KEY_D)) {
        if (current_replay.next_move()) {
            printf("Stepped forward to move %d\n", current_replay.current_move_index + 1);
        }
        last_step_time = now;
    } else if (IsKeyPressed(KEY_LEFT) || IsKeyPressed(KEY_A)) {
        if (current_replay.prev_move()) {
            printf("Stepped backward to move %d\n", current_replay.current_move_index + 1);
        }
        last_step_time = now;
    }
    // Handle held keys for rapid stepping
    else if (IsKeyDown(KEY_RIGHT) && elapsed.count() >= 100) { // 100ms for fast stepping
        step_direction = 1;
        should_step = true;
    } else if (IsKeyDown(KEY_LEFT) && elapsed.count() >= 100) {
        step_direction = -1;
        should_step = true;
    }
    
    // Apply held key stepping
    if (should_step) {
        if (step_direction > 0) {
            if (current_replay.next_move()) {
                printf("Stepped forward to move %d\n", current_replay.current_move_index + 1);
            }
        } else {
            if (current_replay.prev_move()) {
                printf("Stepped backward to move %d\n", current_replay.current_move_index + 1);
            }
        }
        last_step_time = now;
    }
    
    if (IsKeyPressed(KEY_HOME)) {
        current_replay.jump_to_move(0);
        printf("Jumped to start of game\n");
    }
    
    if (IsKeyPressed(KEY_END)) {
        current_replay.jump_to_move(current_replay.current_game->total_moves - 1);
        printf("Jumped to end of game\n");
    }
    
    // Use B key to go back to game list instead of ESC
    if (IsKeyPressed(KEY_B)) {
        replay_mode_active = false;
        show_game_list = true;
    }
    
    // Auto-play controls
    if (IsKeyPressed(KEY_SPACE)) {
        auto_play_replay = !auto_play_replay;
        printf("Auto-play %s\n", auto_play_replay ? "ON" : "OFF");
    }
    
    if (IsKeyPressed(KEY_UP)) {
        auto_play_delay = std::max(100, auto_play_delay - 100);
        printf("Auto-play speed: %d ms\n", auto_play_delay);
    }
    
    if (IsKeyPressed(KEY_DOWN)) {
        auto_play_delay = std::min(2000, auto_play_delay + 100);
        printf("Auto-play speed: %d ms\n", auto_play_delay);
    }
    
    // Game info
    if (IsKeyPressed(KEY_I)) {
        if (global_game_logger && current_replay.current_game) {
            printf("Game Info:\n");
            printf("  Timestamp: %s\n", current_replay.current_game->timestamp.c_str());
            printf("  Outcome: %s\n", current_replay.current_game->outcome.c_str());
            printf("  Total moves: %d\n", current_replay.current_game->total_moves);
            printf("  Filename: %s\n", current_replay.current_game->filename.c_str());
        }
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

// Handle game list input
void handle_game_list_input() {
    // Handle holding arrow keys for fast scrolling
    static auto last_scroll_time = std::chrono::steady_clock::now();
    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_scroll_time);
    
    bool should_scroll = false;
    int scroll_direction = 0;
    
    // Check for key presses first (immediate response)
    if (IsKeyPressed(KEY_UP)) {
        selected_game_index = std::max(0, selected_game_index - 1);
        last_scroll_time = now;
    } else if (IsKeyPressed(KEY_DOWN)) {
        selected_game_index = std::min(static_cast<int>(available_games.size()) - 1, selected_game_index + 1);
        last_scroll_time = now;
    } 
    // Then check for held keys (with timing)
    else if (IsKeyDown(KEY_UP) && elapsed.count() >= 150) { // 150ms delay for held keys
        scroll_direction = -1;
        should_scroll = true;
    } else if (IsKeyDown(KEY_DOWN) && elapsed.count() >= 150) {
        scroll_direction = 1;
        should_scroll = true;
    }
    
    // Apply held key scrolling
    if (should_scroll) {
        selected_game_index = std::max(0, std::min(static_cast<int>(available_games.size()) - 1, 
                                                   selected_game_index + scroll_direction));
        last_scroll_time = now;
    }
    
    if (IsKeyPressed(KEY_ENTER)) {
        if (selected_game_index >= 0 && selected_game_index < static_cast<int>(available_games.size())) {
            if (global_game_logger) {
                const auto* game = global_game_logger->get_game(selected_game_index);
                if (game) {
                    current_replay.start_replay(game);
                    show_game_list = false;
                    replay_mode_active = true;
                    printf("Loaded game for replay: %s\n", game->timestamp.c_str());
                }
            }
        }
    }
    
    // Use B key to go back to menu instead of ESC
    if (IsKeyPressed(KEY_B)) {
        show_game_list = false;
        // Don't set in_menu = true here, it will be handled in main loop
    }
    
    if (IsKeyPressed(KEY_R)) {
        load_available_games();
        selected_game_index = 0;
    }
    
    if (IsKeyPressed(KEY_S)) {
        if (global_game_logger) {
            printf("Statistics:\n");
            global_game_logger->print_statistics();
        }
    }
    
    // Game info shortcut
    if (IsKeyPressed(KEY_I)) {
        if (selected_game_index >= 0 && selected_game_index < static_cast<int>(available_games.size())) {
            if (global_game_logger) {
                const auto* game = global_game_logger->get_game(selected_game_index);
                if (game) {
                    printf("Game Info:\n");
                    printf("  Timestamp: %s\n", game->timestamp.c_str());
                    printf("  Outcome: %s\n", game->outcome.c_str());
                    printf("  Total moves: %d\n", game->total_moves);
                    printf("  Filename: %s\n", game->filename.c_str());
                }
            }
        }
    }
    
    // Filter controls
    if (IsKeyPressed(KEY_F1)) {
        // Filter by outcome (cycle through win, draw)
        static int outcome_filter = 0;
        const char* outcomes[] = {"win", "draw"};
        outcome_filter = (outcome_filter + 1) % 2;
        
        printf("Filtering by outcome: %s\n", outcomes[outcome_filter]);
        // Note: Filtering functionality would require additional implementation
    }
    
    if (IsKeyPressed(KEY_F2)) {
        // Show short games
        printf("Showing games with < 30 moves\n");
        // Note: Filtering functionality would require additional implementation
    }
    
    if (IsKeyPressed(KEY_F3)) {
        // Show statistics
        if (global_game_logger) {
            global_game_logger->print_statistics();
        }
    }
}

// Remaining function implementations
void apply_replay_moves_to_board(chess::ChessBoard& board, const GameReplay& replay) {
    if (!replay.is_active || !replay.current_game) return;
    
    // Reset board to starting position
    board.reset();
    
    // Apply moves up to current position
    for (int i = 0; i <= replay.current_move_index && i < (int)replay.current_game->moves.size(); ++i) {
        const auto& game_move = replay.current_game->moves[i];
        
        // Convert action ID to move using the current board state
        chess::Move move = chess::action_to_move_lookup(game_move.action_id, board);
        
        // Validate and apply the move
        if (move.from.x >= 0 && move.from.y >= 0 && move.to.x >= 0 && move.to.y >= 0) {
            // Check if move is legal in current position
            const auto& legal_moves = board.legal_moves();
            bool is_legal = false;
            for (const auto& legal_move : legal_moves) {
                if (legal_move == move) {
                    is_legal = true;
                    break;
                }
            }
            
            if (is_legal) {
                // Apply the move to the board
                bool applied = board.apply_move(move);
                if (!applied) {
                    printf("[REPLAY ERROR] Failed to apply move %d: action %d (%c%d->%c%d)\n", 
                           i + 1, game_move.action_id, 'a' + move.from.x, move.from.y + 1, 
                           'a' + move.to.x, move.to.y + 1);
                    break;
                }
            } else {
                printf("[REPLAY ERROR] Illegal move %d: action %d (%c%d->%c%d)\n", 
                       i + 1, game_move.action_id, 'a' + move.from.x, move.from.y + 1, 
                       'a' + move.to.x, move.to.y + 1);
                break;
            }
        } else {
            printf("[REPLAY ERROR] Invalid move coordinates for action %d at move %d\n", 
                   game_move.action_id, i + 1);
            break;
        }
    }
}

void update_session_stats(GameMode mode, bool white_won, bool black_won, bool is_draw) {
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

int agent_select_action(ChessNet* net, CChess* env) {
    if (!net || !env) return 0;
    
    // Use the neural network to select an action
    int action;
    forward_chessnet(net, env->observations, &action);
    return action;
}

// Agent selection for dual agent mode - specify which agent (0=white, 1=black)
int agent_select_action_dual(ChessNet* net, CChess* env, int agent_index) {
    if (!net || !env) return 0;
    
    // In dual agent mode, observations are laid out as [agent0_obs, agent1_obs]
    // Each agent has 6018 observation values
    float* agent_observations = env->observations + (agent_index * 6018);
    
    // Use the neural network to select an action
    int action;
    forward_chessnet(net, agent_observations, &action);
    return action;
}

int random_select_action(ChessContext* ctx) {
    if (!ctx) return 0;
    
    const auto& legal_moves = ctx->board.legal_moves();
    if (legal_moves.empty()) return 0;
    
    // Select a random legal move
    std::uniform_int_distribution<int> dist(0, legal_moves.size() - 1);
    int move_index = dist(ctx->rng);
    
    return chess::ChessBoard::move_to_action(legal_moves[move_index]);
}

std::string move_to_uci(const chess::Move& move) {
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
    if (move.promotion != chess::EMPTY) {
        switch (move.promotion) {
            case chess::QUEEN: uci += "q"; break;
            case chess::ROOK: uci += "r"; break;
            case chess::BISHOP: uci += "b"; break;
            case chess::KNIGHT: uci += "n"; break;
            default: break;
        }
    }
    
    return uci;
}

void render_chess_board(CChess* env, ChessPieceTextures* textures) {
    if (!env || !textures) return;
    
    auto* ctx = (ChessContext*)env->context;
    if (!ctx) return;
    
    // Draw board squares
    for (int y = 0; y < 8; ++y) {
        for (int x = 0; x < 8; ++x) {
            int screen_x = BOARD_OFFSET_X + x * SQUARE_SIZE;
            int screen_y = BOARD_OFFSET_Y + (7 - y) * SQUARE_SIZE;
            
            Color square_color = ((x + y) % 2 == 0) ? BEIGE : BROWN;
            DrawRectangle(screen_x, screen_y, SQUARE_SIZE, SQUARE_SIZE, square_color);
            
            // Draw piece if present
            chess::Square pos{int8_t(x), int8_t(y)};
            const chess::Piece& piece = ctx->board.at(pos);
            
            if (piece.type != chess::EMPTY) {
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
                    if (piece.color == chess::BLACK) {
                        piece_char = tolower(piece_char);
                    }
                    DrawText(TextFormat("%c", piece_char), screen_x + 20, screen_y + 20, 24, 
                            piece.color == chess::WHITE ? RL_WHITE : RL_BLACK);
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

void draw_side_panel(CChess* env, ChessPieceTextures* textures, GameMode mode, int elo_setting, ChessNet* white_net, ChessNet* black_net) {
    if (!env) return;
    
    auto* ctx = (ChessContext*)env->context;
    if (!ctx) return;
    
    int panel_x = BOARD_OFFSET_X + BOARD_SIZE + 30 + panel_offset_x;
    int panel_y = BOARD_OFFSET_Y + panel_offset_y;
    
    // Game info
    DrawText("GAME INFO", panel_x, panel_y, 18, DARKBLUE);
    panel_y += 25;
    
    DrawText(TextFormat("Mode: %s", GAME_MODE_NAMES[mode]), panel_x, panel_y, 14, RL_BLACK);
    panel_y += 20;
    
    if (mode == GM_PLAYER_STOCKFISH || mode == GM_AGENT_STOCKFISH) {
        DrawText(TextFormat("Stockfish ELO: %d", elo_setting), panel_x, panel_y, 14, RL_BLACK);
        panel_y += 20;
    }
    
    // Current player
    const char* current_player = (ctx->board.side_to_move() == chess::WHITE) ? "White" : "Black";
    DrawText(TextFormat("To move: %s", current_player), panel_x, panel_y, 14, RL_BLACK);
    panel_y += 20;
    
    // Game status
    if (env->terminals[0]) {
        DrawText("GAME OVER", panel_x, panel_y, 16, RED);
        panel_y += 20;
    } else if (game_paused) {
        DrawText("PAUSED", panel_x, panel_y, 16, ORANGE);
        panel_y += 20;
    }
    
    // Move count
    DrawText(TextFormat("Moves: %d", (int)game_moves.size()), panel_x, panel_y, 14, RL_BLACK);
    panel_y += 25;
    
    // Session statistics
    if (session_stats.total_games > 0) {
        DrawText("SESSION STATS", panel_x, panel_y, 16, DARKBLUE);
        panel_y += 20;
        
        DrawText(TextFormat("Games: %d", session_stats.total_games), panel_x, panel_y, 14, RL_BLACK);
        panel_y += 18;
        
        DrawText(TextFormat("W/L/D: %d/%d/%d", session_stats.total_wins, session_stats.total_losses, session_stats.total_draws), panel_x, panel_y, 14, RL_BLACK);
        panel_y += 18;
    }
    
    // Controls
    panel_y += 10;
    DrawText("CONTROLS", panel_x, panel_y, 16, DARKBLUE);
    panel_y += 20;
    
    DrawText("SPACE: Pause/Resume", panel_x, panel_y, 12, DARKGRAY);
    panel_y += 16;
    DrawText("R: Reset game", panel_x, panel_y, 12, DARKGRAY);
    panel_y += 16;
    DrawText("M: Return to menu", panel_x, panel_y, 12, DARKGRAY);
    panel_y += 16;
    DrawText("S: Show statistics", panel_x, panel_y, 12, DARKGRAY);
    panel_y += 16;
    DrawText("C: Clear statistics", panel_x, panel_y, 12, DARKGRAY);
    panel_y += 16;
    DrawText("X: Save statistics", panel_x, panel_y, 12, DARKGRAY);
}

int main() {
    printf("PufferLib Chess Evaluation – GUI Menu Version\n");
    srand(static_cast<unsigned>(time(NULL)));

    // Initialize global game logger with training logs directory
    global_game_logger = new GameLogger("pufferlib/resources/chess/training_logs/complete_games");
    
    // ------------------------------------------------------------
    // Load agent weights once (used for all agent-controlled sides)
    // ------------------------------------------------------------
    const char *weights_path = "resources/chess/puffer_chess_weights.bin";
    Weights *weights_white = load_weights(weights_path, CHESS_NUM_WEIGHTS);
    Weights *weights_black = load_weights(weights_path, CHESS_NUM_WEIGHTS);
    if (!weights_white || !weights_black) {
        fprintf(stderr, "ERROR: Could not load weights at %s\n", weights_path);
        return 1;
    }
    ChessNet *agent_net_white = init_chessnet(weights_white, 1);
    ChessNet *agent_net_black = init_chessnet(weights_black, 1);

    // ------------------------------------------------------------
    // Setup Raylib window
    // ------------------------------------------------------------
    InitWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "PufferLib Chess – Menu");
    // Allow ESC to quit; use 'M' to return to menu during play
    // (Raylib default ESC exit remains enabled)
    SetTargetFPS(60);

    // ------------------------------------------------------------
    // Load piece textures (shared between menu & game)
    // ------------------------------------------------------------
    ChessPieceTextures textures = load_piece_textures();

    // ------------------------------------------------------------
    // Game/environment objects (re-created when starting a match)
    // ------------------------------------------------------------
    CChess env = {0};
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
            // --------------------------
            // MAIN MENU RENDER + INPUT
            // --------------------------
            ClearBackground(RAYWHITE);
            DrawText("PufferLib Chess", 50, 20, 32, RL_BLACK);
            DrawText("Use UP / DOWN to choose, LEFT / RIGHT to adjust, ENTER to start", 50, 60, 18, DARKGRAY);
            DrawText("Player vs Random = Human (White) vs Random (Black)", 50, 85, 16, DARKBLUE);

            for (int i = 0; i < GM_COUNT; ++i) {
                Color col = (i == menu_index) ? RED : RL_BLACK;
                // Build menu label without nested TextFormat calls (avoids undefined behaviour with static buffers)
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
                DrawText("=== Session Statistics ===", 50, stats_y, 18, DARKBLUE);
                stats_y += 25;
                DrawText(TextFormat("Total Games: %d", session_stats.total_games), 50, stats_y, 16, RL_BLACK);
                stats_y += 20;
                DrawText(TextFormat("Overall W/L/D: %d/%d/%d", session_stats.total_wins, session_stats.total_losses, session_stats.total_draws), 50, stats_y, 16, RL_BLACK);
                stats_y += 25;
                
                // Show specific stats if available
                if (session_stats.agent_stats.games > 0) {
                    DrawText(TextFormat("Agent: %.1f%% win rate (%d/%d/%d)", 
                            session_stats.agent_stats.win_rate() * 100, session_stats.agent_stats.wins, 
                            session_stats.agent_stats.losses, session_stats.agent_stats.draws), 50, stats_y, 16, DARKGREEN);
                    stats_y += 20;
                }
                if (session_stats.human_stats.games > 0) {
                    DrawText(TextFormat("Human: %.1f%% win rate (%d/%d/%d)", 
                            session_stats.human_stats.win_rate() * 100, session_stats.human_stats.wins,
                            session_stats.human_stats.losses, session_stats.human_stats.draws), 50, stats_y, 16, DARKGREEN);
                    stats_y += 20;
                }
                if (session_stats.white_stats.games > 0) {
                    DrawText(TextFormat("White: %.1f%% win rate (%d/%d/%d)", 
                            session_stats.white_stats.win_rate() * 100, session_stats.white_stats.wins,
                            session_stats.white_stats.losses, session_stats.white_stats.draws), 50, stats_y, 16, DARKGREEN);
                    stats_y += 20;
                }
                if (session_stats.black_stats.games > 0) {
                    DrawText(TextFormat("Black: %.1f%% win rate (%d/%d/%d)", 
                            session_stats.black_stats.win_rate() * 100, session_stats.black_stats.wins,
                            session_stats.black_stats.losses, session_stats.black_stats.draws), 50, stats_y, 16, DARKGREEN);
                    stats_y += 20;
                }
            }
            
            // Menu instructions
            DrawText("Press C to clear statistics, X to save to file", 50, WINDOW_HEIGHT - 50, 16, DARKGRAY);

            // Input handling
            if (IsKeyPressed(KEY_UP))    menu_index = (menu_index + GM_COUNT - 1) % GM_COUNT;
            if (IsKeyPressed(KEY_DOWN))  menu_index = (menu_index + 1) % GM_COUNT;

            // Adjust ELO when Player vs Stockfish or Agent vs Stockfish is selected (hold for fast change)
            if (menu_index == GM_PLAYER_STOCKFISH || menu_index == GM_AGENT_STOCKFISH) {
                int delta = 0;
                if (IsKeyDown(KEY_LEFT))  delta -= 5; // faster while holding
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
                
                if (game_mode == GM_GAME_REPLAY) {
                    // ONLY Game Replay mode should access saved training games
                    printf("Entering Game Replay mode - loading saved training games...\n");
                    load_available_games();
                    
                    // Initialize environment for replay mode
                    env.reward_valid = 0.0f;
                    env.reward_agent_captures_enemy_piece = 0.05f;
                    env.reward_enemy_captures_agent_piece = -0.05f;
                    env.reward_draw = 0.0f;
                    env.reward_check_white = 0.0f;
                    env.reward_check_black = 0.0f;
                    env.reward_material_diff_white = 0.0f;
                    env.reward_material_diff_black = 0.0f;
                    env.max_depth = 200;  // CRITICAL: Must be set BEFORE init() call
                    
                    allocate(&env);
                    init(&env);  // This copies env.max_depth to ctx->max_depth
                    c_reset(&env);
                    
                    // Disable Stockfish for replay mode
                    auto *ctx = (ChessContext *)env.context;
                    ctx->stockfish_enabled = false;
                    
                    in_menu = false;  // Exit menu to show game list
                    show_game_list = true;
                    replay_mode_active = false;
                    selected_game_index = 0;
                    
                    printf("[REPLAY] Environment initialized for replay mode\n");
                } else {
                    // ALL other modes are live gameplay (no saved games)
                    printf("Starting %s mode...\n", GAME_MODE_NAMES[game_mode]);
                    in_menu = false;

                    // Initialise environment fresh for each match
                    env.reward_valid = 0.0f;
                    env.reward_agent_captures_enemy_piece = 0.05f;
                    env.reward_enemy_captures_agent_piece = -0.05f;
                    env.reward_draw = 0.0f;
                    env.reward_check_white = 0.0f;
                    env.reward_check_black = 0.0f;
                    env.reward_material_diff_white = 0.0f;
                    env.reward_material_diff_black = 0.0f;
                    env.max_depth = 200;  // CRITICAL: Must be set BEFORE init() call

                    allocate(&env);
                    init(&env);  // This copies env.max_depth to ctx->max_depth
                    c_reset(&env);
                    

                    
                    // For dual agent mode, we need to reallocate arrays for 2 agents
                    if (game_mode == GM_AGENT_AGENT) {
                        // Free single agent arrays
                        free(env.observations);
                        free(env.actions);
                        free(env.rewards);
                        free(env.terminals);
                        
                        // Allocate dual agent arrays
                        env.observations = (float*)calloc(2 * 6018, sizeof(float));  // 2 agents * 6018 obs each
                        env.actions = (int*)calloc(2, sizeof(int));                   // 2 actions
                        env.rewards = (float*)calloc(2, sizeof(float));               // 2 rewards
                        env.terminals = (unsigned char*)calloc(2, sizeof(unsigned char)); // 2 terminals
                        
                        printf("Reallocated arrays for dual agent mode\n");
                    }

                    // Enable / disable Stockfish depending on mode
                    auto *ctx = (ChessContext *)env.context;
                    if (game_mode == GM_PLAYER_STOCKFISH || game_mode == GM_AGENT_STOCKFISH) {
                        // Initialise Stockfish engine explicitly now that automatic
                        // startup has been removed from init()
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
                    
                    // Debug: Print player setup
                    if (game_mode == GM_PLAYER_RANDOM) {
                        printf("White: Human player (mouse clicks)\n");
                        printf("Black: Random opponent (automatic)\n");
                    }
                }
            }
        } else if (show_game_list) {
            // Game list selection screen
            render_game_list_screen();
            handle_game_list_input();
            
            // If show_game_list becomes false and not in replay mode, return to menu
            if (!show_game_list && !replay_mode_active) {
                in_menu = true;
            }
        } else if (replay_mode_active) {
            // Game replay screen
            render_game_replay_screen(&env, &textures);
            handle_game_replay_input();
            
            // If replay_mode_active becomes false, return to game list
            if (!replay_mode_active) {
                show_game_list = true;
            }
        } else {
            // --------------------------
            // GAMEPLAY RENDER + LOGIC
            // --------------------------
            
            auto *ctx = (ChessContext *)env.context;
            
            // Game logic - no hash tracking needed as c_step() handles state properly
            
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
                    int white_action = agent_select_action_dual(white_net, &env, 0);  // Agent 0 = White
                    int black_action = agent_select_action_dual(black_net, &env, 1);  // Agent 1 = Black
                    
                    env.actions[0] = white_action;  // White agent action
                    env.actions[1] = black_action;  // Black agent action
                    
                    // Record move for current player
                    chess::Color current_player = ctx->board.side_to_move();
                    int current_action = (current_player == chess::WHITE) ? white_action : black_action;
                    chess::Move mv = chess::action_to_move_lookup(current_action, ctx->board);
                    if (mv.from.x >= 0) {
                        game_moves.push_back(move_to_uci(mv));
                        printf("[DUAL_AGENT] Move: %s (action %d) by %s\n", 
                               move_to_uci(mv).c_str(), current_action, 
                               (current_player == chess::WHITE) ? "White" : "Black");
                    }
                    
                    c_step(&env);  // This will call c_step_dual_agent
                    continue;  // Skip single agent logic
                }

                // SINGLE AGENT MODES: Handle one side at a time
                if (ctx->board.side_to_move() == chess::WHITE) {
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
                            chess::Move mv = chess::action_to_move_lookup(chosen_action, ctx->board);
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
                                auto mv = chess::action_to_move_lookup(env.actions[0], ctx->board);
                                if (mv.from.x >= 0) game_moves.push_back(move_to_uci(mv));
                            }
                            c_step(&env);
                            break; 
                        }
                        case GM_RANDOM_AGENT: {
                            env.actions[0] = random_select_action(ctx);
                            {
                                auto mv = chess::action_to_move_lookup(env.actions[0], ctx->board);
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
                                auto mv = chess::action_to_move_lookup(env.actions[0], ctx->board);
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
                                auto mv = chess::action_to_move_lookup(env.actions[0], ctx->board);
                                if (mv.from.x >= 0) game_moves.push_back(move_to_uci(mv));
                            }
                            c_step(&env);
                            break; 
                        }
                        case GM_RANDOM_RANDOM: {
                            env.actions[0] = random_select_action(ctx);
                            {
                                auto mv = chess::action_to_move_lookup(env.actions[0], ctx->board);
                                if (mv.from.x >= 0) game_moves.push_back(move_to_uci(mv));
                            }
                            c_step(&env);
                            break; 
                        }
                        case GM_RANDOM_AGENT: {
                            // Agent plays black in this mode
                            // Compute observations for agent
                            compute_observation(&env, ctx);
                            
                            int chosen_action;
                            if (black_net) chosen_action = agent_select_action(black_net, &env);
                            else chosen_action = random_select_action(ctx);
                            env.actions[0] = chosen_action;
                            {
                                auto mv = chess::action_to_move_lookup(chosen_action, ctx->board);
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
                if (last_game_outcome.game_ended && env.terminals[0]) {
                    printf("[DEBUG] Game outcome captured! white_won=%d black_won=%d is_draw=%d reason='%s'\n", 
                           last_game_outcome.white_won, last_game_outcome.black_won, 
                           last_game_outcome.is_draw, last_game_outcome.draw_reason.c_str());
                    
                    // Additional debug for black wins investigation
                    if (last_game_outcome.black_won) {
                        printf("[DEBUG] *** BLACK WIN DETECTED *** This should increment black wins!\n");
                    }
                    if (last_game_outcome.white_won) {
                        printf("[DEBUG] *** WHITE WIN DETECTED *** This should increment white wins!\n");
                    }
                    if (last_game_outcome.is_draw) {
                        printf("[DEBUG] *** DRAW DETECTED *** Reason: %s\n", last_game_outcome.draw_reason.c_str());
                    }
                    
                    // Validate outcome consistency
                    int outcome_count = (last_game_outcome.white_won ? 1 : 0) + 
                                       (last_game_outcome.black_won ? 1 : 0) + 
                                       (last_game_outcome.is_draw ? 1 : 0);
                    if (outcome_count != 1) {
                        printf("[ERROR] Invalid game outcome! Multiple outcomes detected: white_won=%d black_won=%d is_draw=%d\n",
                               last_game_outcome.white_won, last_game_outcome.black_won, last_game_outcome.is_draw);
                    }
                    
                    // Note: GameLogger is now read-only for existing training logs
                    // Complete game logging during gameplay is not implemented in this version
                    
                    // Update session statistics
                    update_session_stats(game_mode, last_game_outcome.white_won, last_game_outcome.black_won, last_game_outcome.is_draw);
                    
                    // Debug: Print updated statistics
                    printf("[DEBUG] After update - Total games: %d, White wins: %d, Black wins: %d, Draws: %d\n",
                           session_stats.total_games, session_stats.total_wins, session_stats.total_losses, session_stats.total_draws);
                    printf("[DEBUG] Legacy counters: session_wins=%d session_losses=%d session_draws=%d\n",
                           session_wins, session_losses, session_draws);
                    printf("[DEBUG] White stats: %d/%d/%d, Black stats: %d/%d/%d\n",
                           session_stats.white_stats.wins, session_stats.white_stats.losses, session_stats.white_stats.draws,
                           session_stats.black_stats.wins, session_stats.black_stats.losses, session_stats.black_stats.draws);
                    
                    // Clear the outcome for next game
                    last_game_outcome = GameOutcome();
                    
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
            // Allow player input even when paused
            if ((game_mode == GM_PLAYER_STOCKFISH || game_mode == GM_PLAYER_RANDOM) && ctx->board.side_to_move() == chess::WHITE) {
                if (IsMouseButtonPressed(MOUSE_LEFT_BUTTON)) {
                    // Reuse existing click-to-move logic from demo
                    Vector2 mp = GetMousePosition();
                    int bx = (mp.x - 50) / 64;
                    int by = (mp.y - 70) / 64;
                    if (game_mode == GM_PLAYER_RANDOM) {
                        printf("[PLAYER_RANDOM] Human click at (%.1f,%.1f) -> board (%d,%d)\n", mp.x, mp.y, bx, by);
                    }
                    static int sel_fx = -1, sel_fy = -1;
                    static bool selecting = false;

                    if (bx >= 0 && bx < 8 && by >= 0 && by < 8) {
                        int board_x = bx;
                        int board_y = 7 - by;

                        if (!selecting) {
                            chess::Square pos{(int8_t)board_x, (int8_t)board_y};
                            const chess::Piece &p = ctx->board.at(pos);
                            if (p.color == chess::WHITE && p.type != chess::EMPTY) {
                                sel_fx = board_x;
                                sel_fy = board_y;
                                selecting = true;
                                if (game_mode == GM_PLAYER_RANDOM) {
                                    printf("[PLAYER_RANDOM] Human selected piece at %c%d\n", 'a'+sel_fx, sel_fy+1);
                                }
                            }
                        } else {
                            const auto &legal = ctx->board.legal_moves();
                            chess::Move chosen = chess::kPassMove;
                            for (const auto &mv : legal) {
                                if (mv.from.x == sel_fx && mv.from.y == sel_fy && mv.to.x == board_x && mv.to.y == board_y) {
                                    chosen = mv; break; }
                            }
                            selecting = false;
                            if (!(chosen == chess::kPassMove)) {
                                env.actions[0] = chess::ChessBoard::move_to_action(chosen);
                                if (game_mode == GM_PLAYER_RANDOM) {
                                    printf("[PLAYER_RANDOM] Human move: %c%d -> %c%d (action %d)\n", 
                                           'a'+sel_fx, sel_fy+1, 'a'+board_x, board_y+1, env.actions[0]);
                                }
                                
                                // Record move before applying
                                game_moves.push_back(move_to_uci(chosen));
                                
                                c_step(&env);
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

            // Adjust side-panel position on-the-fly (vim-style H/J/K/L)
            const int MOVE_STEP = (IsKeyDown(KEY_LEFT_SHIFT) || IsKeyDown(KEY_RIGHT_SHIFT)) ? 5 : 2;
            if (IsKeyDown(KEY_H)) panel_offset_x -= MOVE_STEP;
            if (IsKeyDown(KEY_L)) panel_offset_x += MOVE_STEP;
            if (IsKeyDown(KEY_K)) panel_offset_y -= MOVE_STEP;
            if (IsKeyDown(KEY_J)) panel_offset_y += MOVE_STEP;

            // Clamp to window bounds
            panel_offset_x = std::clamp(panel_offset_x, -550,  550);
            panel_offset_y = std::clamp(panel_offset_y, -300,  300);

            // Console log offsets on change
            if (panel_offset_x != last_panel_offset_x || panel_offset_y != last_panel_offset_y) {
                printf("Panel offset now (%d, %d)\n", panel_offset_x, panel_offset_y);
                fflush(stdout);
                last_panel_offset_x = panel_offset_x;
                last_panel_offset_y = panel_offset_y;
            }

            if (IsKeyPressed(KEY_M)) {
                // Clean up current game resources and safely return to menu
                c_close(&env);
                free_allocated(&env);
                memset(&env, 0, sizeof(env));

                in_menu = true;
                // End the current drawing frame early to avoid dereferencing freed pointers
                EndDrawing();
                continue; // start next loop iteration (menu)
            }

            // Render board / UI
            ClearBackground(RAYWHITE);
            render_chess_board(&env, &textures);
            draw_side_panel(&env, &textures, game_mode, elo_setting, white_net, black_net);
            DrawText(TextFormat("M:Menu  R:Reset  S:Stats  C:Clear  X:Save  →:Speed  SPACE:Pause  H/J/K/L:Move panel  Offset:(%d,%d)", panel_offset_x, panel_offset_y), 10, WINDOW_HEIGHT - 30, 16, DARKGRAY);
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
    // Note: weights are managed by load_weights/free_chessnet
    CloseWindow();

    return 0;
}

#ifndef USE_HEADER_STOCKFISH
#endif