// chess.c
#include "chess.h"
#include "puffernet.h"

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
    
    // Policy head: 256 -> 4674
    Linear *policy_head;   // 256 -> 4674
    
    // Value head: 256 -> 128 -> 1
    Linear *value_head1;   // 256 -> 128
    ReLU   *value_relu;
    Linear *value_head2;   // 128 -> 1

    Multidiscrete *md;     // For action selection (4674 actions)
};

// Calculate total weights for current architecture:
// board_encoder: (1344*512 + 512) + (512*256 + 256) = 688128 + 131328 = 819456
// combiner: (256*256 + 256) = 65792
// policy_head: (256*4674 + 4674) = 1201218
// value_head: (256*128 + 128) + (128*1 + 1) = 32897
// Total: 2119363
#define CHESS_NUM_WEIGHTS 2119363

static ChessNet *init_chessnet(Weights *weights, int num_agents) {
    ChessNet *net = calloc(1, sizeof(ChessNet));
    net->num_agents = num_agents;

    // Board encoder: 1344 -> 512 -> 256
    net->board_enc1  = make_linear(weights, num_agents, 1344, 512);
    net->board_relu1 = make_relu(num_agents, 512);
    net->board_enc2  = make_linear(weights, num_agents, 512, 256);
    net->board_relu2 = make_relu(num_agents, 256);
    
    // Combiner: 256 -> 256 
    net->combiner    = make_linear(weights, num_agents, 256, 256);
    net->comb_relu   = make_relu(num_agents, 256);
    
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
    
    // Policy head: 256 -> 4674
    linear(net->policy_head, net->comb_relu->output);
    
    // Select action using argmax
    argmax_multidiscrete(net->md, net->policy_head->output, actions);
    
    // Value head (optional, result ignored for now)
    linear(net->value_head1, net->comb_relu->output);
    relu(net->value_relu, net->value_head1->output);
    linear(net->value_head2, net->value_relu->output);
}

// -----------------------------------------------------------------------------
// Demo / simple performance test harness
// -----------------------------------------------------------------------------

int main() {
    // Environment parameters corresponding to chess.ini defaults
    CChess env = {
        .reward_valid = 0.0f,
        .reward_invalid = -0.1f,
        .reward_agent_captures_enemy_piece = 0.05f,
        .reward_enemy_captures_agent_piece = -0.05f,
        .reward_win = 1.0f,
        .reward_draw = 0.0f,
        .reward_loss = -1.0f,
    };

    // Allocate buffers and reset environment
    allocate(&env);
    init(&env);
    c_reset(&env);

    printf("Chess environment initialized\n");
    printf("Observation size: 1344 (21 channels × 8×8)\n");
    printf("Action space: 4674 possible moves\n");
    printf("Expected weights for network: %d\n", CHESS_NUM_WEIGHTS);
    
    // For now, just run environment without network to test basic functionality
    const int total_steps = 1000;
    int games_played = 0;
    
    for (int i = 0; i < total_steps; i++) {
        // Use simple random actions for now
        env.actions[0] = i % 4674;

        // Step environment
        c_step(&env);
        if (env.terminals[0]) {
            games_played++;
            c_reset(&env);
        }
    }

    printf("Completed %d steps, %d games played\n", total_steps, games_played);

    // Cleanup
    c_close(&env);
    free_allocated(&env);
    return 0;
}