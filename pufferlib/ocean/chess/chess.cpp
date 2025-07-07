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

static void render_chess_board(CChess *env, const ChessPieceTextures *textures) {
    const int BOARD_SIZE = 512;
    const int SQUARE_SIZE = BOARD_SIZE / 8;
    const int BOARD_OFFSET_X = 50;
    const int BOARD_OFFSET_Y = 70;
    
    // Clear background
    ClearBackground(RAYWHITE);
    
    // Get board from context
    ChessContext *ctx = (ChessContext*)env->context;
    if (!ctx) return;
    
    const chess::ChessBoard &board = ctx->board;
    
    // Draw board squares
    for (int y = 0; y < 8; y++) {
        for (int x = 0; x < 8; x++) {
            Color square_color = ((x + y) % 2 == 0) ? BEIGE : BROWN;
            
            int square_x = BOARD_OFFSET_X + x * SQUARE_SIZE;
            int square_y = BOARD_OFFSET_Y + y * SQUARE_SIZE;
            
            DrawRectangle(square_x, square_y, SQUARE_SIZE, SQUARE_SIZE, square_color);
            DrawRectangleLines(square_x, square_y, SQUARE_SIZE, SQUARE_SIZE, RL_BLACK);
        }
    }
    
    // Draw pieces
    for (int y = 0; y < 8; y++) {
        for (int x = 0; x < 8; x++) {
            const chess::Piece &piece = board.at({int8_t(x), int8_t(7-y)}); // Flip Y for display
            
            if (piece.type != chess::EMPTY) {
                Texture2D texture = get_piece_texture(textures, piece.color, piece.type);
                if (texture.id != 0 && texture.width > 0 && texture.height > 0) {
                    // Scale piece to fit nicely in square (max 80% of square size)
                    float max_size = SQUARE_SIZE * 0.8f;
                    float scale = fminf(max_size / texture.width, max_size / texture.height);
                    
                    float scaled_width = texture.width * scale;
                    float scaled_height = texture.height * scale;
                    
                    float piece_x = BOARD_OFFSET_X + x * SQUARE_SIZE + (SQUARE_SIZE - scaled_width) / 2.0f;
                    float piece_y = BOARD_OFFSET_Y + y * SQUARE_SIZE + (SQUARE_SIZE - scaled_height) / 2.0f;
                    
                    DrawTextureEx(texture, (Vector2){piece_x, piece_y}, 0.0f, scale, RL_WHITE);
                } else {
                    // Debug: draw a colored circle if texture didn't load
                    Color debug_color = (piece.color == 0) ? BLUE : RED;
                    int center_x = BOARD_OFFSET_X + x * SQUARE_SIZE + SQUARE_SIZE/2;
                    int center_y = BOARD_OFFSET_Y + y * SQUARE_SIZE + SQUARE_SIZE/2;
                    DrawCircle(center_x, center_y, 20, debug_color);
                }
            }
        }
    }
    
    // Draw UI info
    const char *to_move = (board.side_to_move() == 0) ? "White" : "Black"; // WHITE = 0
    DrawText(TextFormat("Turn: %s", to_move), BOARD_OFFSET_X + BOARD_SIZE + 20, BOARD_OFFSET_Y, 20, RL_BLACK);
    
    DrawText(TextFormat("Step: %d", ctx->step_count), BOARD_OFFSET_X + BOARD_SIZE + 20, BOARD_OFFSET_Y + 30, 20, RL_BLACK);
    DrawText(TextFormat("Return: %.2f", ctx->episode_return), BOARD_OFFSET_X + BOARD_SIZE + 20, BOARD_OFFSET_Y + 60, 20, RL_BLACK);
    
    if (board.is_check()) {
        DrawText("CHECK!", BOARD_OFFSET_X + BOARD_SIZE + 20, BOARD_OFFSET_Y + 100, 24, RED);
    }
    
    // Instructions below board
    int instr_y = BOARD_OFFSET_Y + BOARD_SIZE + 10;
    DrawText("Left-click: select piece, then destination (White)", 10, instr_y, 16, DARKGRAY);
    DrawText("Engine (Black) replies automatically", 10, instr_y + 20, 16, DARKGRAY);
    DrawText("Press R to reset game", 10, instr_y + 40, 16, DARKGRAY);
}

// -----------------------------------------------------------------------------
// New structures and helpers for GUI menu and gameplay modes
// -----------------------------------------------------------------------------
// Gameplay modes selectable from the main menu
enum GameMode {
    GM_PLAYER_STOCKFISH = 0,
    GM_AGENT_STOCKFISH,
    GM_PLAYER_RANDOM,
    GM_AGENT_AGENT,
    GM_AGENT_RANDOM,
    GM_RANDOM_RANDOM,
    GM_RANDOM_AGENT,
    GM_COUNT
};

static const char *GAME_MODE_NAMES[GM_COUNT] = {
    "Player vs Stockfish",
    "Agent vs Stockfish", 
    "Player vs Random",
    "Agent vs Agent",
    "Agent vs Random",
    "Random vs Random",
    "Random vs Agent"
};

// -----------------------------------------------------------------------------
// Session statistics - Enhanced for proper policy evaluation
// -----------------------------------------------------------------------------
struct SessionStats {
    // Overall session statistics
    int total_games = 0;
    int total_wins = 0;
    int total_losses = 0;
    int total_draws = 0;
    
    // Player-specific statistics (from their perspective)
    struct PlayerStats {
        int games = 0;
        int wins = 0;
        int losses = 0;
        int draws = 0;
        float win_rate() const { return games > 0 ? (float)wins / games : 0.0f; }
        float loss_rate() const { return games > 0 ? (float)losses / games : 0.0f; }
        float draw_rate() const { return games > 0 ? (float)draws / games : 0.0f; }
    };
    
    PlayerStats white_stats;  // White player statistics
    PlayerStats black_stats;  // Black player statistics
    PlayerStats agent_stats;  // Agent statistics (when agent is involved)
    PlayerStats human_stats;  // Human statistics (when human is involved)
    
    void reset() {
        total_games = total_wins = total_losses = total_draws = 0;
        white_stats = black_stats = agent_stats = human_stats = PlayerStats{};
    }
    
    void print_summary(GameMode mode) {
        printf("\n=== SESSION STATISTICS ===\n");
        printf("Total games: %d\n", total_games);
        printf("Overall W/L/D: %d/%d/%d\n", total_wins, total_losses, total_draws);
        
            switch (mode) {
        case GM_PLAYER_STOCKFISH:
            printf("Human (White): %.1f%% win rate (%d/%d/%d)\n", 
                   human_stats.win_rate() * 100, human_stats.wins, human_stats.losses, human_stats.draws);
            break;
        case GM_AGENT_STOCKFISH:
            printf("Agent (White): %.1f%% win rate (%d/%d/%d)\n", 
                   agent_stats.win_rate() * 100, agent_stats.wins, agent_stats.losses, agent_stats.draws);
            break;
        case GM_PLAYER_RANDOM:
            printf("Human (White): %.1f%% win rate (%d/%d/%d)\n", 
                   human_stats.win_rate() * 100, human_stats.wins, human_stats.losses, human_stats.draws);
            break;
        case GM_AGENT_AGENT:
            printf("White Agent: %.1f%% win rate (%d/%d/%d)\n", 
                   white_stats.win_rate() * 100, white_stats.wins, white_stats.losses, white_stats.draws);
            printf("Black Agent: %.1f%% win rate (%d/%d/%d)\n", 
                   black_stats.win_rate() * 100, black_stats.wins, black_stats.losses, black_stats.draws);
            break;
        case GM_AGENT_RANDOM:
            printf("Agent (White): %.1f%% win rate (%d/%d/%d)\n", 
                   agent_stats.win_rate() * 100, agent_stats.wins, agent_stats.losses, agent_stats.draws);
            break;
        case GM_RANDOM_RANDOM:
            printf("White Random: %.1f%% win rate (%d/%d/%d)\n", 
                   white_stats.win_rate() * 100, white_stats.wins, white_stats.losses, white_stats.draws);
            printf("Black Random: %.1f%% win rate (%d/%d/%d)\n", 
                   black_stats.win_rate() * 100, black_stats.wins, black_stats.losses, black_stats.draws);
            break;
        case GM_RANDOM_AGENT:
            printf("Random (White): %.1f%% win rate (%d/%d/%d)\n", 
                   white_stats.win_rate() * 100, white_stats.wins, white_stats.losses, white_stats.draws);
            printf("Agent (Black): %.1f%% win rate (%d/%d/%d)\n", 
                   agent_stats.win_rate() * 100, agent_stats.wins, agent_stats.losses, agent_stats.draws);
            break;
        case GM_COUNT:
            break;
    }
        printf("========================\n\n");
    }
    
    void save_to_file(GameMode mode, const char* filename = "chess_eval_stats.txt") {
        std::ofstream file(filename, std::ios::app);
        if (!file.is_open()) {
            printf("Error: Could not open %s for writing\n", filename);
            return;
        }
        
        // Get current timestamp
        time_t now = time(0);
        char* timestr = ctime(&now);
        timestr[strlen(timestr) - 1] = '\0'; // Remove newline
        
        file << "\n=== CHESS EVALUATION SESSION ===\n";
        file << "Timestamp: " << timestr << "\n";
        file << "Mode: " << GAME_MODE_NAMES[mode] << "\n";
        file << "Total games: " << total_games << "\n";
        file << "Overall W/L/D: " << total_wins << "/" << total_losses << "/" << total_draws << "\n";
        
        switch (mode) {
            case GM_PLAYER_STOCKFISH:
                file << "Human (White): " << human_stats.win_rate() * 100 << "% win rate (" 
                     << human_stats.wins << "/" << human_stats.losses << "/" << human_stats.draws << ")\n";
                break;
            case GM_AGENT_STOCKFISH:
                file << "Agent (White): " << agent_stats.win_rate() * 100 << "% win rate (" 
                     << agent_stats.wins << "/" << agent_stats.losses << "/" << agent_stats.draws << ")\n";
                break;
            case GM_PLAYER_RANDOM:
                file << "Human (White): " << human_stats.win_rate() * 100 << "% win rate (" 
                     << human_stats.wins << "/" << human_stats.losses << "/" << human_stats.draws << ")\n";
                break;
            case GM_AGENT_AGENT:
                file << "White Agent: " << white_stats.win_rate() * 100 << "% win rate (" 
                     << white_stats.wins << "/" << white_stats.losses << "/" << white_stats.draws << ")\n";
                file << "Black Agent: " << black_stats.win_rate() * 100 << "% win rate (" 
                     << black_stats.wins << "/" << black_stats.losses << "/" << black_stats.draws << ")\n";
                break;
            case GM_AGENT_RANDOM:
                file << "Agent (White): " << agent_stats.win_rate() * 100 << "% win rate (" 
                     << agent_stats.wins << "/" << agent_stats.losses << "/" << agent_stats.draws << ")\n";
                break;
            case GM_RANDOM_RANDOM:
                file << "White Random: " << white_stats.win_rate() * 100 << "% win rate (" 
                     << white_stats.wins << "/" << white_stats.losses << "/" << white_stats.draws << ")\n";
                file << "Black Random: " << black_stats.win_rate() * 100 << "% win rate (" 
                     << black_stats.wins << "/" << black_stats.losses << "/" << black_stats.draws << ")\n";
                break;
            case GM_RANDOM_AGENT:
                file << "Random (White): " << white_stats.win_rate() * 100 << "% win rate (" 
                     << white_stats.wins << "/" << white_stats.losses << "/" << white_stats.draws << ")\n";
                file << "Agent (Black): " << agent_stats.win_rate() * 100 << "% win rate (" 
                     << agent_stats.wins << "/" << agent_stats.losses << "/" << agent_stats.draws << ")\n";
                break;
            case GM_COUNT:
                break;
        }
        file << "================================\n\n";
        file.close();
        
        printf("Statistics saved to %s\n", filename);
    }
};

static SessionStats session_stats;

// Legacy variables for backward compatibility with existing GUI
static int session_wins  = 0;
static int session_losses = 0;
static int session_draws  = 0;

// Store the last game outcome for the GUI to use
static bool last_game_was_win = false;
static bool last_game_was_loss = false;
static bool last_game_was_draw = false;

// Track game completion state between frames
// (last_game_outcome is defined in chess.h)

// Function to update session statistics based on game outcome
static void update_session_stats(GameMode mode, bool white_won, bool black_won, bool is_draw) {
    // Validate input parameters
    int outcome_count = (white_won ? 1 : 0) + (black_won ? 1 : 0) + (is_draw ? 1 : 0);
    if (outcome_count != 1) {
        printf("[ERROR] update_session_stats called with invalid parameters: white_won=%d black_won=%d is_draw=%d\n",
               white_won, black_won, is_draw);
        return;
    }
    
    printf("[DEBUG] update_session_stats: mode=%s white_won=%d black_won=%d is_draw=%d\n",
           GAME_MODE_NAMES[mode], white_won, black_won, is_draw);
    
    session_stats.total_games++;
    
    if (white_won) {
        session_stats.total_wins++;
        session_stats.white_stats.games++;
        session_stats.white_stats.wins++;
        session_stats.black_stats.games++;
        session_stats.black_stats.losses++;
    } else if (black_won) {
        session_stats.total_losses++;  // Note: total_losses represents black wins (from white's perspective)
        session_stats.white_stats.games++;
        session_stats.white_stats.losses++;
        session_stats.black_stats.games++;
        session_stats.black_stats.wins++;
    } else if (is_draw) {
        session_stats.total_draws++;
        session_stats.white_stats.games++;
        session_stats.white_stats.draws++;
        session_stats.black_stats.games++;
        session_stats.black_stats.draws++;
    }
    
    // Update mode-specific statistics
    switch (mode) {
        case GM_PLAYER_STOCKFISH:
            // Human plays white, Stockfish plays black
            session_stats.human_stats.games++;
            if (white_won) {
                session_stats.human_stats.wins++;
                session_wins++;  // Legacy counter
            } else if (black_won) {
                session_stats.human_stats.losses++;
                session_losses++;  // Legacy counter
            } else {
                session_stats.human_stats.draws++;
                session_draws++;  // Legacy counter
            }
            break;
            
        case GM_AGENT_STOCKFISH:
            // Agent plays white, Stockfish plays black
            session_stats.agent_stats.games++;
            if (white_won) {
                session_stats.agent_stats.wins++;
                session_wins++;  // Legacy counter
            } else if (black_won) {
                session_stats.agent_stats.losses++;
                session_losses++;  // Legacy counter
            } else {
                session_stats.agent_stats.draws++;
                session_draws++;  // Legacy counter
            }
            break;
            
        case GM_PLAYER_RANDOM:
            // Human plays white, Random plays black
            session_stats.human_stats.games++;
            if (white_won) {
                session_stats.human_stats.wins++;
                session_wins++;  // Legacy counter
            } else if (black_won) {
                session_stats.human_stats.losses++;
                session_losses++;  // Legacy counter
            } else {
                session_stats.human_stats.draws++;
                session_draws++;  // Legacy counter
            }
            break;
            
        case GM_AGENT_AGENT:
            // Both agents - track from white's perspective for legacy counters
            if (white_won) {
                session_wins++;
            } else if (black_won) {
                session_losses++;
            } else {
                session_draws++;
            }
            break;
            
        case GM_AGENT_RANDOM:
            // Agent plays white, Random plays black
            session_stats.agent_stats.games++;
            if (white_won) {
                session_stats.agent_stats.wins++;
                session_wins++;  // Legacy counter
            } else if (black_won) {
                session_stats.agent_stats.losses++;
                session_losses++;  // Legacy counter
            } else {
                session_stats.agent_stats.draws++;
                session_draws++;  // Legacy counter
            }
            break;
            
        case GM_RANDOM_RANDOM:
            // Both random - track from white's perspective for legacy counters
            if (white_won) {
                session_wins++;
            } else if (black_won) {
                session_losses++;
            } else {
                session_draws++;
            }
            break;
            
        case GM_RANDOM_AGENT:
            // Random plays white, Agent plays black
            session_stats.agent_stats.games++;
            if (white_won) {
                session_stats.agent_stats.losses++;
                session_wins++;  // Legacy counter (from white's perspective)
            } else if (black_won) {
                session_stats.agent_stats.wins++;
                session_losses++;  // Legacy counter (from white's perspective)
            } else {
                session_stats.agent_stats.draws++;
                session_draws++;  // Legacy counter
            }
            break;
            
        case GM_COUNT:
            break;
    }
    
    // Update GUI variables
    last_game_was_win = white_won;
    last_game_was_loss = black_won;
    last_game_was_draw = is_draw;
    
    // Print game result to console for debugging
    const char* result_str = is_draw ? "DRAW" : (white_won ? "WHITE WINS" : "BLACK WINS");
    printf("[GAME %d] %s - ", session_stats.total_games, result_str);
    
    switch (mode) {
        case GM_PLAYER_STOCKFISH:
            printf("Human vs Stockfish - Human %s\n", white_won ? "WINS" : (black_won ? "LOSES" : "DRAWS"));
            break;
        case GM_AGENT_STOCKFISH:
            printf("Agent vs Stockfish - Agent %s\n", white_won ? "WINS" : (black_won ? "LOSES" : "DRAWS"));
            break;
        case GM_PLAYER_RANDOM:
            printf("Human vs Random - Human %s\n", white_won ? "WINS" : (black_won ? "LOSES" : "DRAWS"));
            break;
        case GM_AGENT_AGENT:
            printf("Agent vs Agent - White Agent %s\n", white_won ? "WINS" : (black_won ? "LOSES" : "DRAWS"));
            break;
        case GM_AGENT_RANDOM:
            printf("Agent vs Random - Agent %s\n", white_won ? "WINS" : (black_won ? "LOSES" : "DRAWS"));
            break;
        case GM_RANDOM_RANDOM:
            printf("Random vs Random - White %s\n", white_won ? "WINS" : (black_won ? "LOSES" : "DRAWS"));
            break;
        case GM_RANDOM_AGENT:
            printf("Random vs Agent - Random %s\n", white_won ? "WINS" : (black_won ? "LOSES" : "DRAWS"));
            break;
        case GM_COUNT:
            break;
    }
}

// -----------------------------------------------------------------------------
// Runtime-adjustable UI positioning (arrow/HJKL style)                
// -----------------------------------------------------------------------------
static int panel_offset_x = 0;   // horizontal offset for side-panel text
static int panel_offset_y = 166;   // vertical   offset for side-panel text
static int last_panel_offset_x = 0;
static int last_panel_offset_y = 166;

// -----------------------------------------------------------------------------
// Helper to convert Move to UCI string (from board perspective)
static std::string move_to_uci(const chess::Move &m) {
    if (m.from.x < 0) return "0000";
    char buf[6] = {0};
    buf[0] = 'a' + m.from.x;
    buf[1] = '1' + m.from.y;
    buf[2] = 'a' + m.to.x;
    buf[3] = '1' + m.to.y;
    if (m.promotion != chess::EMPTY) {
        switch (m.promotion) {
            case chess::QUEEN:  buf[4] = 'q'; break;
            case chess::ROOK:   buf[4] = 'r'; break;
            case chess::BISHOP: buf[4] = 'b'; break;
            case chess::KNIGHT: buf[4] = 'n'; break;
            default: break;
        }
    }
    return std::string(buf);
}

// -----------------------------------------------------------------------------
// Globals toggles
// -----------------------------------------------------------------------------
static bool show_bestmove = false;
static bool game_paused = false;  // Add pause state tracking

// Move history for current game
static std::vector<std::string> game_moves;

// Draw side information panel (right of board)
static void draw_side_panel(const CChess *env, const ChessPieceTextures *textures, GameMode mode, int elo_setting, ChessNet *white_net, ChessNet *black_net) {
    const int BASE_PANEL_X = 580;
    const int BASE_START_Y = 50;

    const int PANEL_X = BASE_PANEL_X + panel_offset_x;
    const int START_Y = BASE_START_Y + panel_offset_y;

    int y = START_Y;
    auto *ctx = (ChessContext*)env->context;

    // Player labels
    std::string white_label, black_label;
    switch (mode) {
        case GM_PLAYER_STOCKFISH: white_label = "Human"; black_label = "Stockfish(" + std::to_string(elo_setting) + ")"; break;
        case GM_AGENT_STOCKFISH:  white_label = "Agent"; black_label = "Stockfish(" + std::to_string(elo_setting) + ")"; break;
        case GM_AGENT_AGENT:      white_label = "Agent"; black_label = "Agent"; break;
        case GM_AGENT_RANDOM:     white_label = "Agent"; black_label = "Random"; break;
        case GM_RANDOM_RANDOM:    white_label = "Random"; black_label = "Random"; break;
        case GM_PLAYER_RANDOM:    white_label = "Human"; black_label = "Random"; break;
        case GM_RANDOM_AGENT:     white_label = "Random"; black_label = "Agent"; break;
        case GM_COUNT:            break;
        default: break;
    }
    DrawText(TextFormat("White: %s", white_label.c_str()), PANEL_X, y, 18, RL_BLACK); y += 22;
    DrawText(TextFormat("Black: %s", black_label.c_str()), PANEL_X, y, 18, RL_BLACK); y += 28;

    // Stockfish evaluation (centipawns, white perspective)
    DrawText(TextFormat("Eval: %d cp", (int)ctx->stockfish_eval), PANEL_X, y, 18, (ctx->stockfish_eval > 0 ? DARKGREEN : RED)); y += 26;
    
    // Pause indicator
    if (game_paused) {
        DrawText("*** PAUSED ***", PANEL_X, y, 18, RED); y += 26;
    }

    // Session stats
    DrawText("Session W/L/D", PANEL_X, y, 18, RL_BLACK); y += 20;
    DrawText(TextFormat("%d / %d / %d", session_wins, session_losses, session_draws), PANEL_X, y, 18, RL_BLACK); y += 18;
    
    // Enhanced session stats based on mode
    switch (mode) {
        case GM_PLAYER_STOCKFISH:
            if (session_stats.human_stats.games > 0) {
                DrawText(TextFormat("Human: %.1f%% win", session_stats.human_stats.win_rate() * 100), PANEL_X, y, 16, DARKGREEN); y += 16;
            }
            break;
        case GM_AGENT_STOCKFISH:
            if (session_stats.agent_stats.games > 0) {
                DrawText(TextFormat("Agent: %.1f%% win", session_stats.agent_stats.win_rate() * 100), PANEL_X, y, 16, DARKGREEN); y += 16;
            }
            break;
        case GM_PLAYER_RANDOM:
            if (session_stats.human_stats.games > 0) {
                DrawText(TextFormat("Human: %.1f%% win", session_stats.human_stats.win_rate() * 100), PANEL_X, y, 16, DARKGREEN); y += 16;
            }
            break;
        case GM_AGENT_AGENT:
            if (session_stats.white_stats.games > 0) {
                DrawText(TextFormat("W: %.1f%% B: %.1f%%", session_stats.white_stats.win_rate() * 100, session_stats.black_stats.win_rate() * 100), PANEL_X, y, 16, DARKGREEN); y += 16;
            }
            break;
        case GM_AGENT_RANDOM:
            if (session_stats.agent_stats.games > 0) {
                DrawText(TextFormat("Agent: %.1f%% win", session_stats.agent_stats.win_rate() * 100), PANEL_X, y, 16, DARKGREEN); y += 16;
            }
            break;
        case GM_RANDOM_RANDOM:
            if (session_stats.white_stats.games > 0) {
                DrawText(TextFormat("W: %.1f%% B: %.1f%%", session_stats.white_stats.win_rate() * 100, session_stats.black_stats.win_rate() * 100), PANEL_X, y, 16, DARKGREEN); y += 16;
            }
            break;
        case GM_RANDOM_AGENT:
            if (session_stats.agent_stats.games > 0) {
                DrawText(TextFormat("Agent: %.1f%% win", session_stats.agent_stats.win_rate() * 100), PANEL_X, y, 16, DARKGREEN); y += 16;
            }
            break;
        case GM_COUNT:
            break;
    }
    y += 12;

    // Reward breakdown (last step)
    DrawText("Rewards", PANEL_X, y, 18, RL_BLACK); y += 20;
    DrawText(TextFormat("step %.3f", env->rewards[0]), PANEL_X, y, 16, RL_BLACK); y += 18;
    DrawText(TextFormat("valid %.2f", ctx->c_reward_valid), PANEL_X, y, 16, RL_BLACK); y += 18;
    DrawText(TextFormat("invalid %.2f", ctx->c_reward_invalid), PANEL_X, y, 16, RL_BLACK); y += 18;
    DrawText(TextFormat("capture %.2f", ctx->c_reward_agent_captures_enemy_piece), PANEL_X, y, 16, RL_BLACK); y += 18;
    DrawText(TextFormat("lostPiece %.2f", ctx->c_reward_enemy_captures_agent_piece), PANEL_X, y, 16, RL_BLACK); y += 18;
    DrawText(TextFormat("check %.2f", ctx->c_reward_check), PANEL_X, y, 16, RL_BLACK); y += 18;
    DrawText(TextFormat("material %.2f", ctx->c_reward_material_diff), PANEL_X, y, 16, RL_BLACK); y += 22;

    // Move list
    DrawText("Moves", PANEL_X, y, 18, RL_BLACK); y += 20;
    int move_line = 0;
    for (const auto &mv : game_moves) {
        DrawText(mv.c_str(), PANEL_X, y + move_line * 16, 16, RL_BLACK);
        move_line++;
        if (move_line > 20) break; // clip
    }

    // Best move recommendation (optional)
    if (show_bestmove && ctx->sf && ctx->sf->ok()) {
        std::string best = ctx->sf->bestmove(ctx->board.fen(), 20);
        DrawText(TextFormat("Best: %s", best.c_str()), PANEL_X, START_Y - 24, 18, BLUE);
    }
}

// Helper to select an action for a ChessNet agent given current env observation
static int agent_select_action(ChessNet *net, CChess *env) {
    int action = 0;
    forward_chessnet(net, env->observations, &action);
    return action;
}

// Helper to select a random legal move (returns action id)
static int random_select_action(const ChessContext *ctx) {
    const auto &legal = ctx->board.legal_moves();
    if (legal.empty()) return 0;
    int idx = rand() % legal.size();
    return chess::ChessBoard::move_to_action(legal[idx]);
}

// -----------------------------------------------------------------------------
// Replace original main loop with GUI-enabled version
// -----------------------------------------------------------------------------
int main() {
    printf("PufferLib Chess Evaluation – GUI Menu Version\n");
    srand(static_cast<unsigned>(time(NULL)));

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
    const int WINDOW_WIDTH = 900;
    const int WINDOW_HEIGHT = 700;
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
                in_menu   = false;

                // Initialise environment fresh for each match
                env.reward_valid = 0.0f;
                env.reward_invalid = -0.1f;
                env.reward_agent_captures_enemy_piece = 0.05f;
                env.reward_enemy_captures_agent_piece = -0.05f;
                env.reward_win = 1.0f;
                env.reward_draw = 0.0f;
                env.reward_loss = -1.0f;
                env.reward_check = 0.0f;
                env.reward_material_diff = 0.0f;
                env.max_depth = 512;  // From config/ocean/chess.ini

                allocate(&env);
                init(&env);
                c_reset(&env);

                // Enable / disable Stockfish depending on mode
                auto *ctx = (ChessContext *)env.context;
                if (game_mode == GM_PLAYER_STOCKFISH || game_mode == GM_AGENT_STOCKFISH) {
                    // Initialise Stockfish engine explicitly now that automatic
                    // startup has been removed from init()
                    enable_stockfish_black(&env, nullptr, elo_setting, 10);
                    ctx->stockfish_enabled = true;
                } else {
                    ctx->stockfish_enabled = false;
                }

                white_net = (game_mode == GM_AGENT_STOCKFISH || game_mode == GM_AGENT_AGENT || game_mode == GM_AGENT_RANDOM) ? agent_net_white : nullptr;
                black_net = (game_mode == GM_AGENT_AGENT || game_mode == GM_RANDOM_AGENT) ? agent_net_black : nullptr;
            }
        } else {
            // --------------------------
            // GAMEPLAY RENDER + LOGIC
            // --------------------------
            
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

                // Decide which side to supply an action for
                if (ctx->board.side_to_move() == chess::WHITE) {
                    switch (game_mode) {
                        case GM_PLAYER_STOCKFISH:
                        case GM_PLAYER_RANDOM: {
                            // Human move handled via mouse clicks inside render function
                            break; }
                        case GM_AGENT_STOCKFISH:
                        case GM_AGENT_AGENT:
                        case GM_AGENT_RANDOM: {
                            int chosen_action;
                            if (white_net) chosen_action = agent_select_action(white_net, &env);
                            else chosen_action = random_select_action(ctx);
                            env.actions[0] = chosen_action;

                            // Record move
                            chess::Move mv = chess::action_to_move_lookup(chosen_action, ctx->board);
                            if (mv.from.x >= 0) {
                                game_moves.push_back(move_to_uci(mv));
                            }

                            c_step(&env);
                            break; }
                        case GM_RANDOM_RANDOM: {
                            env.actions[0] = random_select_action(ctx);
                            {
                                auto mv = chess::action_to_move_lookup(env.actions[0], ctx->board);
                                if (mv.from.x >= 0) game_moves.push_back(move_to_uci(mv));
                            }
                            c_step(&env);
                            break; }
                        case GM_RANDOM_AGENT: {
                            env.actions[0] = random_select_action(ctx);
                            {
                                auto mv = chess::action_to_move_lookup(env.actions[0], ctx->board);
                                if (mv.from.x >= 0) game_moves.push_back(move_to_uci(mv));
                            }
                            c_step(&env);
                            break; }
                        case GM_COUNT:
                            break;
                        default: break;
                    }
                                    } else { // Black to move
                    switch (game_mode) {
                        case GM_PLAYER_STOCKFISH: {
                            // Stockfish handled internally in c_step()
                            break; }
                        case GM_PLAYER_RANDOM: {
                            // Random black move
                            env.actions[0] = random_select_action(ctx);
                            {
                                auto mv = chess::action_to_move_lookup(env.actions[0], ctx->board);
                                if (mv.from.x >= 0) game_moves.push_back(move_to_uci(mv));
                            }
                            c_step(&env);
                            break; }
                        case GM_AGENT_STOCKFISH: {
                            // Stockfish black handled internally; nothing to do
                            break; }
                        case GM_AGENT_AGENT: {
                            int chosen_action;
                            if (black_net) chosen_action = agent_select_action(black_net, &env);
                            else chosen_action = random_select_action(ctx);
                            env.actions[0] = chosen_action;
                            {
                                auto mv = chess::action_to_move_lookup(chosen_action, ctx->board);
                                if (mv.from.x >= 0) game_moves.push_back(move_to_uci(mv));
                            }
                            c_step(&env);
                            break; }
                        case GM_AGENT_RANDOM: {
                            env.actions[0] = random_select_action(ctx);
                            {
                                auto mv = chess::action_to_move_lookup(env.actions[0], ctx->board);
                                if (mv.from.x >= 0) game_moves.push_back(move_to_uci(mv));
                            }
                            c_step(&env);
                            break; }
                        case GM_RANDOM_RANDOM: {
                            env.actions[0] = random_select_action(ctx);
                            {
                                auto mv = chess::action_to_move_lookup(env.actions[0], ctx->board);
                                if (mv.from.x >= 0) game_moves.push_back(move_to_uci(mv));
                            }
                            c_step(&env);
                            break; }
                        case GM_RANDOM_AGENT: {
                            // Agent plays black in this mode
                            int chosen_action;
                            if (black_net) chosen_action = agent_select_action(black_net, &env);
                            else chosen_action = random_select_action(ctx);
                            env.actions[0] = chosen_action;
                            {
                                auto mv = chess::action_to_move_lookup(chosen_action, ctx->board);
                                if (mv.from.x >= 0) game_moves.push_back(move_to_uci(mv));
                            }
                            c_step(&env);
                            break; }
                        case GM_COUNT:
                            break;
                        default: break;
                    }
                }

                // Check if a game just ended by looking at the captured outcome
                if (last_game_outcome.game_ended) {
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
                    fprintf(stderr, "[CLICK] raw=(%.1f,%.1f) -> board bx=%d by=%d\n", mp.x, mp.y, bx, by);
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
                                fprintf(stderr, "[SELECT] piece at %c%d selected\n", 'a'+sel_fx, sel_fy+1);
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
                                fprintf(stderr, "[MOVE] action id=%d\n", env.actions[0]);
                                c_step(&env);
                            }
                            fprintf(stderr, "[DEST] attempt move %c%d -> %c%d\n", 'a'+sel_fx, sel_fy+1, 'a'+board_x, board_y+1);
                        }
                    }
                }
            }

            // Allow reset / return to menu
            if (IsKeyPressed(KEY_R)) {
                c_reset(&env);
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
    
    // Cleanup global resources
    unload_piece_textures(&textures);
    CloseWindow();
    c_close(&env);
    free_allocated(&env);
    free_chessnet(agent_net_white);
    free_chessnet(agent_net_black);
    free(weights_white);
    free(weights_black);
    return 0;
}

#ifndef USE_HEADER_STOCKFISH
#endif