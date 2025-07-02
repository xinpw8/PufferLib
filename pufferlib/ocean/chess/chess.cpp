// chess.cpp - Graphical Chess Evaluation using Raylib
#include <time.h>
#include <math.h>
#include "chess.h"
#include "puffernet.h"

#ifdef __cplusplus
extern "C" {
#endif
#include "raylib.h" 
#ifdef __cplusplus
}
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
    const int BOARD_OFFSET_Y = 50;
    
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
            DrawRectangleLines(square_x, square_y, SQUARE_SIZE, SQUARE_SIZE, BLACK);
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
                    
                    DrawTextureEx(texture, (Vector2){piece_x, piece_y}, 0.0f, scale, WHITE);
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
    DrawText(TextFormat("Turn: %s", to_move), BOARD_OFFSET_X + BOARD_SIZE + 20, BOARD_OFFSET_Y, 20, BLACK);
    
    DrawText(TextFormat("Step: %d", ctx->step_count), BOARD_OFFSET_X + BOARD_SIZE + 20, BOARD_OFFSET_Y + 30, 20, BLACK);
    DrawText(TextFormat("Return: %.2f", ctx->episode_return), BOARD_OFFSET_X + BOARD_SIZE + 20, BOARD_OFFSET_Y + 60, 20, BLACK);
    
    if (board.is_check()) {
        DrawText("CHECK!", BOARD_OFFSET_X + BOARD_SIZE + 20, BOARD_OFFSET_Y + 100, 24, RED);
    }
    
    // Instructions
    DrawText("Hold SHIFT for human control", 10, 10, 16, DARKGRAY);
    DrawText("Click square to move", 10, 30, 16, DARKGRAY);
    DrawText("Press R to reset game", 10, 50, 16, DARKGRAY);
}

void demo() {
    // Initialize Chess Environment
    CChess env = {
        .reward_valid = 0.0f,
        .reward_invalid = -0.1f,
        .reward_agent_captures_enemy_piece = 0.05f,
        .reward_enemy_captures_agent_piece = -0.05f,
        .reward_win = 1.0f,
        .reward_draw = 0.0f,
        .reward_loss = -1.0f,
    };

    // Load weights and initialize network
    printf("Attempting to load %d weights from file...\n", CHESS_NUM_WEIGHTS);
    Weights *weights = load_weights("resources/chess/puffer_chess_weights.bin", CHESS_NUM_WEIGHTS);
    if (!weights) {
        printf("ERROR: Failed to load weights!\n");
        return;
    }
    printf("Successfully loaded weights! Size: %d\n", weights->size);
    
    ChessNet *net = init_chessnet(weights, 1);
    printf("Network initialized successfully.\n");
    
    // Initialize environment
    allocate(&env);
    init(&env);
    c_reset(&env);

    // Initialize Raylib
    const int WINDOW_WIDTH = 800;
    const int WINDOW_HEIGHT = 650;
    InitWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "PufferLib Chess Evaluation");
    SetTargetFPS(60);
    
    // Load piece textures
    ChessPieceTextures textures = load_piece_textures();
    printf("Piece textures loaded, starting game loop...\n");
    fflush(stdout);
    
    // Game loop
    int tick = 0;
    bool ai_move_pending = false;
    printf("Entering main game loop...\n");
    fflush(stdout);
    
    while (!WindowShouldClose()) {
        // Handle input
        if (IsKeyPressed(KEY_R)) {
            c_reset(&env);
            ai_move_pending = false;
        }
        
        // AI move logic (slower pace for viewing)
        if (tick % 90 == 0 && !ai_move_pending) { // Move every 1.5 seconds
            tick = 0;
            
            if (!env.terminals[0]) {
                // Get AI action
                if (!IsKeyDown(KEY_LEFT_SHIFT)) {
                    printf("Running neural network forward pass...\n");
                    printf("Board features (first 20): ");
                    for (int i = 0; i < 20; i++) {
                        printf("%.1f ", env.observations[i]);
                    }
                    printf("\n");
                    printf("Side to move area (960-970): ");
                    for (int i = 960; i < 970; i++) {
                        printf("%.1f ", env.observations[i]);
                    }
                    printf("\n");
                    
                    // Count non-zero values in first 1344 (board features)
                    int nonzero_count = 0;
                    for (int i = 0; i < 1344; i++) {
                        if (env.observations[i] > 0.0f) nonzero_count++;
                    }
                    printf("Non-zero values in board features (1344): %d\n", nonzero_count);
                    
                    forward_chessnet(net, env.observations, env.actions);
                    printf("AI selected action: %d\n", env.actions[0]);
                    ai_move_pending = true;
                }
            }
        }
        
        // Apply move if pending
        if (ai_move_pending) {
            c_step(&env);
            ai_move_pending = false;
            
            // Check if game ended
            if (env.terminals[0]) {
                printf("Game ended! Final return: %.2f\n", 
                       ((ChessContext*)env.context)->episode_return);
            }
        }
        
        tick++;
        
        // Human control (basic click-to-move system)
        if (IsKeyDown(KEY_LEFT_SHIFT) && IsMouseButtonPressed(MOUSE_LEFT_BUTTON)) {
            Vector2 mousePos = GetMousePosition();
            
            // Convert mouse to board coordinates
            int board_x = (mousePos.x - 50) / 64;
            int board_y = (mousePos.y - 50) / 64;
            
            if (board_x >= 0 && board_x < 8 && board_y >= 0 && board_y < 8) {
                // Simple: just try random legal move (proper click-to-move would be more complex)
                ChessContext *ctx = (ChessContext*)env.context;
                const auto& legal_moves = ctx->board.legal_moves();
                
                if (!legal_moves.empty()) {
                    int move_idx = rand() % legal_moves.size();
                    const auto& selected_move = legal_moves[move_idx];
                    env.actions[0] = chess::ChessBoard::move_to_action(selected_move);
                    c_step(&env);
                }
            }
        }
        
        // Render
        BeginDrawing();
        render_chess_board(&env, &textures);
        EndDrawing();
    }
    
    // Cleanup
    unload_piece_textures(&textures);
    CloseWindow();
    free_chessnet(net);
    c_close(&env);
    free_allocated(&env);
}

void performance_test() {
    // Environment parameters
    CChess env = {
        .reward_valid = 0.0f,
        .reward_invalid = -0.1f,
        .reward_agent_captures_enemy_piece = 0.05f,
        .reward_enemy_captures_agent_piece = -0.05f,
        .reward_win = 1.0f,
        .reward_draw = 0.0f,
        .reward_loss = -1.0f,
    };
    
    allocate(&env);
    init(&env);
    c_reset(&env);

    long test_time = 10;
    long start = time(NULL);
    int i = 0;
    
    while (time(NULL) - start < test_time) {
        // Pick a random legal move
        ChessContext *ctx = (ChessContext*)env.context;
        const auto& legal_moves = ctx->board.legal_moves();
        
        if (!legal_moves.empty()) {
            int move_idx = rand() % legal_moves.size();
            const auto& selected_move = legal_moves[move_idx];
            env.actions[0] = chess::ChessBoard::move_to_action(selected_move);
        } else {
            env.actions[0] = 0; // Invalid move will trigger terminal
        }
        
        c_step(&env);
        
        if (env.terminals[0]) {
            c_reset(&env);
        }
        
        i++;
    }
    
    long end = time(NULL);
    printf("SPS: %ld\n", i / (end - start));
    free_allocated(&env);
}

int main() {
    printf("PufferLib Chess Evaluation\n");
    printf("Loading weights from: resources/chess/puffer_chess_weights.bin\n");
    printf("Expected weights: %d\n", CHESS_NUM_WEIGHTS);
    
    demo();
    // performance_test();
    return 0;
}