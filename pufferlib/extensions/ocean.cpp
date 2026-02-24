// ocean.cpp - Environment-specific encoder/decoder models for Ocean environments
// Separated from models.cpp for cleaner organization
// NOTE: This file is included directly into pufferlib.cpp inside namespace pufferlib

// Snake encoder: one-hot encode observations then linear
class SnakeEncoder : public Encoder {
    public:
        torch::nn::Linear linear{nullptr};
        int input;
        int hidden;
        int num_classes;

    SnakeEncoder(int input, int hidden, int num_classes = 8)
        : input(input), hidden(hidden), num_classes(num_classes) {
        linear = register_module("linear", torch::nn::Linear(
            torch::nn::LinearOptions(input * num_classes, hidden).bias(false)));
        torch::nn::init::orthogonal_(linear->weight, std::sqrt(2.0));
    }

    Tensor forward(Tensor x) override {
        // x is [B, input] with values 0-7
        int64_t B = x.size(0);
        auto target_dtype = linear->weight.dtype();
        // One-hot encode: [B, input] -> [B, input, num_classes]
        Tensor onehot = torch::one_hot(x.to(torch::kLong), num_classes).to(target_dtype);
        // Flatten: [B, input * num_classes]
        onehot = onehot.view({B, -1});
        return linear->forward(onehot);
    }
};

// G2048 encoder: embeddings + 3 linear layers with GELU
// Matches Python: value_embed(obs) + pos_embed -> flatten -> encoder MLP
class G2048Encoder : public Encoder {
    public:
        torch::nn::Embedding value_embed{nullptr};
        torch::nn::Embedding pos_embed{nullptr};
        torch::nn::Linear linear1{nullptr};
        torch::nn::Linear linear2{nullptr};
        torch::nn::Linear linear3{nullptr};
        int input;
        int hidden;
        static constexpr int embed_dim = 3;  // ceil(33^0.25) = 3
        static constexpr int num_grid_cells = 16;
        static constexpr int num_obs = num_grid_cells * embed_dim;  // 48

    G2048Encoder(int input, int hidden)
        : input(input), hidden(hidden) {
        // Embeddings for tile values and positions
        value_embed = register_module("value_embed", torch::nn::Embedding(18, embed_dim));
        pos_embed = register_module("pos_embed", torch::nn::Embedding(num_grid_cells, embed_dim));

        // Encoder MLP: num_obs -> 2*hidden -> hidden -> hidden
        linear1 = register_module("linear1", torch::nn::Linear(
            torch::nn::LinearOptions(num_obs, 2*hidden).bias(false)));
        torch::nn::init::orthogonal_(linear1->weight, std::sqrt(2.0));

        linear2 = register_module("linear2", torch::nn::Linear(
            torch::nn::LinearOptions(2*hidden, hidden).bias(false)));
        torch::nn::init::orthogonal_(linear2->weight, std::sqrt(2.0));

        linear3 = register_module("linear3", torch::nn::Linear(
            torch::nn::LinearOptions(hidden, hidden).bias(false)));
        torch::nn::init::orthogonal_(linear3->weight, std::sqrt(2.0));
    }

    Tensor forward(Tensor x) override {
        // x is (B, 16) uint8 tile values
        auto B = x.size(0);
        auto target_dtype = linear1->weight.dtype();

        // value_embed(obs) -> (B, 16, embed_dim)
        auto value_obs = value_embed->forward(x.to(torch::kLong)).to(target_dtype);

        // pos_embed.weight expanded to (B, 16, embed_dim)
        auto pos_obs = pos_embed->weight.unsqueeze(0).expand({B, num_grid_cells, embed_dim}).to(target_dtype);

        // grid_obs = (value_obs + pos_obs).flatten(1) -> (B, 48)
        auto grid_obs = (value_obs + pos_obs).flatten(1);

        // Encoder MLP
        auto h = torch::gelu(linear1->forward(grid_obs));
        h = torch::gelu(linear2->forward(h));
        h = torch::gelu(linear3->forward(h));
        return h;
    }
};

class SimpleG2048Encoder : public Encoder {
    public:
        torch::nn::Embedding value_embed{nullptr};
        torch::nn::Embedding pos_embed{nullptr};
        torch::nn::Linear linear1{nullptr};
        int input;
        int hidden;
        static constexpr int embed_dim = 3;  // ceil(33^0.25) = 3
        static constexpr int num_grid_cells = 16;
        static constexpr int num_obs = num_grid_cells * embed_dim;  // 48

    SimpleG2048Encoder(int input, int hidden)
        : input(input), hidden(hidden) {
        // Embeddings for tile values and positions
        value_embed = register_module("value_embed", torch::nn::Embedding(18, embed_dim));
        pos_embed = register_module("pos_embed", torch::nn::Embedding(num_grid_cells, embed_dim));

        linear1 = register_module("linear1", torch::nn::Linear(
            torch::nn::LinearOptions(num_obs, hidden).bias(false)));
        torch::nn::init::orthogonal_(linear1->weight, std::sqrt(2.0));
    }

    Tensor forward(Tensor x) override {
        // x is (B, 16) uint8 tile values
        auto B = x.size(0);
        auto target_dtype = linear1->weight.dtype();

        // value_embed(obs) -> (B, 16, embed_dim)
        auto value_obs = value_embed->forward(x.to(torch::kLong)).to(target_dtype);

        // pos_embed.weight expanded to (B, 16, embed_dim)
        auto pos_obs = pos_embed->weight.unsqueeze(0).expand({B, num_grid_cells, embed_dim}).to(target_dtype);

        // grid_obs = (value_obs + pos_obs).flatten(1) -> (B, 48)
        auto grid_obs = (value_obs + pos_obs).flatten(1);

        return linear1->forward(grid_obs);
    }
};

// NMMO3 encoder: Conv2d map processing + embedding for player discrete + projection
class NMMO3Encoder : public Encoder {
    public:
        // Multi-hot encoding factors and offsets
        torch::nn::Conv2d conv1{nullptr};
        torch::nn::Conv2d conv2{nullptr};
        torch::nn::Embedding player_embed{nullptr};
        torch::nn::Linear proj{nullptr};
        Tensor offsets{nullptr};
        int input;
        int hidden;

    NMMO3Encoder(int input, int hidden)
        : input(input), hidden(hidden) {
        // factors = [4, 4, 17, 5, 3, 5, 5, 5, 7, 4], sum = 59
        // Map processing: Conv2d(59, 128, 5, stride=3) -> ReLU -> Conv2d(128, 128, 3, stride=1) -> Flatten
        conv1 = register_module("conv1", torch::nn::Conv2d(
            torch::nn::Conv2dOptions(59, 128, 5).stride(3).bias(true)));
        torch::nn::init::orthogonal_(conv1->weight, std::sqrt(2.0));
        torch::nn::init::constant_(conv1->bias, 0.0);

        conv2 = register_module("conv2", torch::nn::Conv2d(
            torch::nn::Conv2dOptions(128, 128, 3).stride(1).bias(true)));
        torch::nn::init::orthogonal_(conv2->weight, std::sqrt(2.0));
        torch::nn::init::constant_(conv2->bias, 0.0);

        // Player discrete encoder: Embedding(128, 32) -> Flatten
        // Input is 47 discrete values, output is 47*32 = 1504
        player_embed = register_module("player_embed", torch::nn::Embedding(128, 32));

        // Projection: Linear(1817, hidden) -> ReLU
        // Input: (B, 59, 11, 15)
        // After conv1(5, stride=3): (11-5)/3+1=3, (15-5)/3+1=4 -> (B, 128, 3, 4)
        // After conv2(3, stride=1): (3-3)/1+1=1, (4-3)/1+1=2 -> (B, 128, 1, 2)
        // Flatten: 128*1*2 = 256
        // player_discrete: 47*32 = 1504
        // player continuous (same 47 values): 47
        // reward: 10
        // Total: 256 + 1504 + 47 + 10 = 1817
        proj = register_module("proj", torch::nn::Linear(
            torch::nn::LinearOptions(1817, hidden).bias(true)));
        torch::nn::init::orthogonal_(proj->weight, std::sqrt(2.0));
        torch::nn::init::constant_(proj->bias, 0.0);

        // Register offsets buffer for multi-hot encoding
        // factors = [4, 4, 17, 5, 3, 5, 5, 5, 7, 4]
        // offsets = [0, 4, 8, 25, 30, 33, 38, 43, 48, 55]
        std::vector<int64_t> offset_vals = {0, 4, 8, 25, 30, 33, 38, 43, 48, 55};
        offsets = register_buffer("offsets",
            torch::tensor(offset_vals, torch::kInt64).view({1, 10, 1, 1}));
    }

    Tensor forward(Tensor x) override {
        int64_t B = x.size(0);
        auto device = x.device();
        auto target_dtype = conv1->weight.dtype();

        // Split observations: map (1650), player (47), reward (10)
        Tensor ob_map = x.narrow(1, 0, 11*15*10).view({B, 11, 15, 10});
        Tensor ob_player = x.narrow(1, 11*15*10, 47);
        Tensor ob_reward = x.narrow(1, 11*15*10 + 47, 10);

        // Multi-hot encoding for map
        // ob_map: (B, 11, 15, 10) -> permute to (B, 10, 11, 15)
        Tensor map_perm = ob_map.permute({0, 3, 1, 2}).to(torch::kInt64);
        // Add offsets: codes = map_perm + offsets
        Tensor codes = map_perm + offsets.to(device);

        // Create multi-hot buffer and scatter
        Tensor map_buf = torch::zeros({B, 59, 11, 15}, torch::TensorOptions().dtype(target_dtype).device(device));
        map_buf.scatter_(1, codes.to(torch::kInt32), 1.0f);

        // Conv layers
        Tensor map_out = torch::relu(conv1->forward(map_buf));
        map_out = conv2->forward(map_out);
        map_out = map_out.flatten(1);  // (B, 256)

        // Player discrete embedding
        Tensor player_discrete = player_embed->forward(ob_player.to(torch::kInt64)).to(target_dtype);
        player_discrete = player_discrete.flatten(1);  // (B, 1504)

        // Concatenate: map_out + player_discrete + player_continuous + reward
        Tensor obs = torch::cat({map_out, player_discrete, ob_player.to(target_dtype), ob_reward.to(target_dtype)}, 1);

        // Projection with ReLU
        obs = torch::relu(proj->forward(obs));
        return obs;
    }
};

// NMMO3 decoder: LayerNorm -> fused logits+value
class NMMO3Decoder : public Decoder {
    public:
        torch::nn::LayerNorm layer_norm{nullptr};
        torch::nn::Linear linear{nullptr};
        int hidden;
        int output;

    NMMO3Decoder(int hidden, int output)
        : hidden(hidden), output(output) {
        layer_norm = register_module("layer_norm", torch::nn::LayerNorm(
            torch::nn::LayerNormOptions({hidden})));

        linear = register_module("linear", torch::nn::Linear(
            torch::nn::LinearOptions(hidden, output + 1).bias(true)));
        torch::nn::init::orthogonal_(linear->weight, 0.01);
        torch::nn::init::constant_(linear->bias, 0.0);
    }

    std::tuple<Logits, Tensor> forward(Tensor h) override {
        Tensor x = layer_norm->forward(h);
        Tensor out = linear->forward(x);
        Tensor logits = out.narrow(-1, 0, output);
        Tensor value = out.narrow(-1, output, 1);
        return {Logits{logits, Tensor()}, value.squeeze(-1)};
    }
};

// Drive encoder: ego/partner/road encoders with max pooling
// Two modes:
//   use_fused_kernel=true:  FC -> Max (fused kernel, no intermediate layer)
//   use_fused_kernel=false: Linear -> LayerNorm -> Linear -> Max (original torch)
class DriveEncoder : public Encoder {
    public:
        // Ego encoder: Linear -> ReLU -> Linear (no max pooling, single point)
        torch::nn::Linear ego_linear1{nullptr};
        torch::nn::Linear ego_linear2{nullptr};

        // Road encoder weights - fused mode: single FC layer
        Tensor road_W{nullptr};
        Tensor road_b{nullptr};
        // Road encoder modules - torch mode: Linear -> LayerNorm -> Linear
        torch::nn::Linear road_linear1{nullptr};
        torch::nn::LayerNorm road_ln{nullptr};
        torch::nn::Linear road_linear2{nullptr};

        // Partner encoder weights - fused mode: single FC layer
        Tensor partner_W{nullptr};
        Tensor partner_b{nullptr};
        // Partner encoder modules - torch mode: Linear -> LayerNorm -> Linear
        torch::nn::Linear partner_linear1{nullptr};
        torch::nn::LayerNorm partner_ln{nullptr};
        torch::nn::Linear partner_linear2{nullptr};

        // Shared embedding
        torch::nn::Linear shared_linear{nullptr};
        int input;
        int hidden;
        bool use_fused_kernel;

    DriveEncoder(int input, int hidden, bool use_fused_kernel = true)
        : input(128), hidden(hidden), use_fused_kernel(use_fused_kernel) {

        // Ego encoder: 7 -> 128 -> 128 (Linear -> ReLU -> Linear)
        ego_linear1 = register_module("ego_linear1", torch::nn::Linear(
            torch::nn::LinearOptions(7, 128).bias(true)));
        torch::nn::init::orthogonal_(ego_linear1->weight, std::sqrt(2.0));
        torch::nn::init::constant_(ego_linear1->bias, 0.0);
        ego_linear2 = register_module("ego_linear2", torch::nn::Linear(
            torch::nn::LinearOptions(128, 128).bias(true)));
        torch::nn::init::orthogonal_(ego_linear2->weight, std::sqrt(2.0));
        torch::nn::init::constant_(ego_linear2->bias, 0.0);

        if (use_fused_kernel) {
            // Fused mode: single FC -> Max (no intermediate layer)
            // Road: 13 -> 128 (6 continuous + 7 one-hot)
            road_W = register_parameter("road_W", torch::empty({128, 13}));
            road_b = register_parameter("road_b", torch::zeros({128}));
            torch::nn::init::orthogonal_(road_W, std::sqrt(2.0));

            // Partner: 7 -> 128
            partner_W = register_parameter("partner_W", torch::empty({128, 7}));
            partner_b = register_parameter("partner_b", torch::zeros({128}));
            torch::nn::init::orthogonal_(partner_W, std::sqrt(2.0));
        } else {
            // Torch mode: Linear -> LayerNorm -> Linear -> Max
            // Road: 13 -> 128 -> 128
            road_linear1 = register_module("road_linear1", torch::nn::Linear(
                torch::nn::LinearOptions(13, 128).bias(true)));
            torch::nn::init::orthogonal_(road_linear1->weight, std::sqrt(2.0));
            torch::nn::init::constant_(road_linear1->bias, 0.0);
            road_ln = register_module("road_ln", torch::nn::LayerNorm(
                torch::nn::LayerNormOptions({128})));
            road_linear2 = register_module("road_linear2", torch::nn::Linear(
                torch::nn::LinearOptions(128, 128).bias(true)));
            torch::nn::init::orthogonal_(road_linear2->weight, std::sqrt(2.0));
            torch::nn::init::constant_(road_linear2->bias, 0.0);

            // Partner: 7 -> 128 -> 128
            partner_linear1 = register_module("partner_linear1", torch::nn::Linear(
                torch::nn::LinearOptions(7, 128).bias(true)));
            torch::nn::init::orthogonal_(partner_linear1->weight, std::sqrt(2.0));
            torch::nn::init::constant_(partner_linear1->bias, 0.0);
            partner_ln = register_module("partner_ln", torch::nn::LayerNorm(
                torch::nn::LayerNormOptions({128})));
            partner_linear2 = register_module("partner_linear2", torch::nn::Linear(
                torch::nn::LinearOptions(128, 128).bias(true)));
            torch::nn::init::orthogonal_(partner_linear2->weight, std::sqrt(2.0));
            torch::nn::init::constant_(partner_linear2->bias, 0.0);
        }

        // Shared embedding: 3*128 -> hidden
        shared_linear = register_module("shared_linear", torch::nn::Linear(
            torch::nn::LinearOptions(3*128, hidden).bias(true)));
        torch::nn::init::orthogonal_(shared_linear->weight, std::sqrt(2.0));
        torch::nn::init::constant_(shared_linear->bias, 0.0);
    }

    Tensor forward(Tensor x) override {
        int64_t B = x.size(0);
        auto target_dtype = ego_linear1->weight.dtype();
        x = x.to(target_dtype);

        // Split observations: ego (7), partner (441), road (1400)
        Tensor ego_obs = x.narrow(1, 0, 7);
        Tensor partner_obs = x.narrow(1, 7, 63*7);
        Tensor road_obs = x.narrow(1, 7 + 63*7, 200*7);

        // Ego encoding: Linear -> ReLU -> Linear (single point, no max)
        Tensor ego_features = ego_linear2->forward(torch::relu(ego_linear1->forward(ego_obs)));

        // Partner encoding
        Tensor partner_objects = partner_obs.view({B, 63, 7}).contiguous();
        Tensor partner_features;
        if (use_fused_kernel) {
            // Fused FC -> Max kernel
            partner_features = FCMax::apply(partner_objects, partner_W, partner_b)[0];
        } else {
            // Torch: Linear -> LayerNorm -> Linear -> Max
            auto h = partner_linear1->forward(partner_objects);  // (B, 63, 128)
            h = partner_ln->forward(h);
            h = partner_linear2->forward(h);  // (B, 63, 128)
            partner_features = std::get<0>(h.max(1));  // (B, 128)
        }

        // Road encoding with one-hot
        Tensor road_objects = road_obs.view({B, 200, 7});
        Tensor road_continuous = road_objects.narrow(2, 0, 6);
        Tensor road_categorical = road_objects.narrow(2, 6, 1).squeeze(2);
        Tensor road_onehot = torch::one_hot(road_categorical.to(torch::kInt64), 7).to(x.dtype());
        Tensor road_combined = torch::cat({road_continuous, road_onehot}, 2).contiguous();  // (B, 200, 13)

        Tensor road_features;
        if (use_fused_kernel) {
            // Fused FC -> Max kernel
            road_features = FCMax::apply(road_combined, road_W, road_b)[0];
        } else {
            // Torch: Linear -> LayerNorm -> Linear -> Max
            auto h = road_linear1->forward(road_combined);  // (B, 200, 128)
            h = road_ln->forward(h);
            h = road_linear2->forward(h);  // (B, 200, 128)
            road_features = std::get<0>(h.max(1));  // (B, 128)
        }

        // Concatenate and shared embedding: GELU -> Linear -> ReLU
        Tensor concat_features = torch::cat({ego_features, road_features, partner_features}, 1);
        Tensor embedding = torch::relu(shared_linear->forward(torch::gelu(concat_features)));
        return embedding;
    }
};

// G2048 decoder: separate policy and value heads, cat + narrow for contiguous output
class G2048Decoder : public Decoder {
    public:
        torch::nn::Linear dec_linear1{nullptr};
        torch::nn::Linear dec_linear2{nullptr};
        torch::nn::Linear val_linear1{nullptr};
        torch::nn::Linear val_linear2{nullptr};
        int hidden;
        int output;

    G2048Decoder(int hidden, int output)
        : hidden(hidden), output(output) {
        // Decoder head: hidden -> hidden -> num_atns
        dec_linear1 = register_module("dec_linear1", torch::nn::Linear(
            torch::nn::LinearOptions(hidden, hidden).bias(false)));
        torch::nn::init::orthogonal_(dec_linear1->weight, std::sqrt(2.0));

        dec_linear2 = register_module("dec_linear2", torch::nn::Linear(
            torch::nn::LinearOptions(hidden, output).bias(false)));
        torch::nn::init::orthogonal_(dec_linear2->weight, 0.01);

        // Value head: hidden -> hidden -> 1
        val_linear1 = register_module("val_linear1", torch::nn::Linear(
            torch::nn::LinearOptions(hidden, hidden).bias(false)));
        torch::nn::init::orthogonal_(val_linear1->weight, std::sqrt(2.0));

        val_linear2 = register_module("val_linear2", torch::nn::Linear(
            torch::nn::LinearOptions(hidden, 1).bias(false)));
        torch::nn::init::orthogonal_(val_linear2->weight, 1.0);
    }

    std::tuple<Logits, Tensor> forward(Tensor h) override {
        // Policy head
        Tensor logits = torch::gelu(dec_linear1->forward(h));
        logits = dec_linear2->forward(logits);

        // Value head
        Tensor value = torch::gelu(val_linear1->forward(h));
        value = val_linear2->forward(value);

        // Cat and narrow for contiguous outputs
        Tensor out = torch::cat({logits, value}, 1).contiguous();
        logits = out.narrow(1, 0, output);
        value = out.narrow(1, output, 1);

        return {Logits{logits, Tensor()}, value.squeeze(1)};
    }
};

// Chess encoder: spatial CNN with geometric priors (adapted from ChessSeven)
// Obs layout (1082 bytes per player):
//   0-767:    12 piece planes × 64 squares (uint8, 0/255)
//   768-769:  side to move (2-byte one-hot)
//   770-785:  castling rights (16-byte one-hot)
//   786-850:  en passant (65-byte one-hot)
//   851-852:  phase (2-byte one-hot: piece selection vs dest selection)
//   853-916:  selected piece (64 bytes, per-square)
//   917-980:  valid pieces mask (64 bytes)
//   981-1044: valid destinations mask (64 bytes)
//   1045-1076: valid promotions (32 bytes)
//   1077-1081: scalars (self_check, opp_check, rule50, repetition, pass_valid)
class ChessEncoder : public Encoder {
    public:
        // CNN pipeline: 1×1 → channel projection → 3×3 depthwise + residual
        nn::Conv2d square_embed{nullptr};   // (spatial_in, square_dim, 1×1)
        nn::Conv2d channel_proj{nullptr};   // (square_dim, proj_dim, 1×1)
        nn::Conv2d spatial_mix{nullptr};    // (proj_dim, proj_dim, 3×3, groups=proj_dim)

        // Categorical embeddings
        nn::Embedding side_embed{nullptr};     // (2, embed_dim/2)
        nn::Embedding castle_embed{nullptr};   // (16, embed_dim)
        nn::Embedding ep_embed{nullptr};       // (65, embed_dim)
        nn::Embedding phase_embed{nullptr};    // (2, embed_dim/2)

        // Final projection to hidden_size
        nn::Linear proj{nullptr};

        // Pre-computed geometric feature planes (registered buffer)
        Tensor square_geo_planes;  // (1, 4, 8, 8)

        int input, hidden;

        static constexpr int SQUARE_DIM = 64;
        static constexpr int PROJ_DIM = 8;
        static constexpr int EMBED_DIM = 32;
        // 12 piece planes + selected_piece + valid_pieces + valid_dests = 15 from obs + 4 geometric
        static constexpr int SPATIAL_IN = 19;

    ChessEncoder(int input, int hidden) : input(input), hidden(hidden) {
        // CNN pipeline (adapted from ChessSeven's depthwise separable approach)
        square_embed = register_module("square_embed", nn::Conv2d(
            nn::Conv2dOptions(SPATIAL_IN, SQUARE_DIM, 1)));
        nn::init::orthogonal_(square_embed->weight, std::sqrt(2.0));
        nn::init::constant_(square_embed->bias, 0.0);

        channel_proj = register_module("channel_proj", nn::Conv2d(
            nn::Conv2dOptions(SQUARE_DIM, PROJ_DIM, 1)));
        nn::init::orthogonal_(channel_proj->weight, std::sqrt(2.0));
        nn::init::constant_(channel_proj->bias, 0.0);

        spatial_mix = register_module("spatial_mix", nn::Conv2d(
            nn::Conv2dOptions(PROJ_DIM, PROJ_DIM, 3).padding(1).groups(PROJ_DIM)));
        nn::init::orthogonal_(spatial_mix->weight, std::sqrt(2.0));
        nn::init::constant_(spatial_mix->bias, 0.0);

        // Categorical embeddings
        side_embed = register_module("side_embed", nn::Embedding(2, EMBED_DIM / 2));
        castle_embed = register_module("castle_embed", nn::Embedding(16, EMBED_DIM));
        ep_embed = register_module("ep_embed", nn::Embedding(65, EMBED_DIM));
        phase_embed = register_module("phase_embed", nn::Embedding(2, EMBED_DIM / 2));

        // Projection: board_flat(512) + promos(32) + embeddings(96) + scalars(5) = 645
        int board_flat = 64 * PROJ_DIM;  // = 512
        int total_features = board_flat + 32 + 3 * EMBED_DIM + 5;  // = 645
        proj = register_module("proj", nn::Linear(
            nn::LinearOptions(total_features, hidden).bias(true)));
        nn::init::orthogonal_(proj->weight, std::sqrt(2.0));
        nn::init::constant_(proj->bias, 0.0);

        // Pre-compute geometric feature planes (diagonal, anti-diagonal, center distance, square color)
        auto sqs = torch::arange(64, torch::kFloat32);
        auto r = torch::div(sqs, 8, "floor");
        auto f = torch::fmod(sqs, 8);
        auto diag = (r + f) / 14.0;
        auto anti = (r - f + 7) / 14.0;
        auto cdist = (torch::where(r < 4, 3 - r, r - 4) +
                      torch::where(f < 4, 3 - f, f - 4)) / 6.0;
        auto sq_color = ((r + f).to(torch::kInt64) % 2).to(torch::kFloat32);
        square_geo_planes = register_buffer("square_geo_planes",
            torch::stack({diag, anti, cdist, sq_color}, 0).view({1, 4, 8, 8}));
    }

    Tensor forward(Tensor x) override {
        int64_t B = x.size(0);
        auto target_dtype = square_embed->weight.dtype();

        // 1. Extract spatial features and reshape to (B, C, 8, 8)
        // Piece planes: 12 channels
        Tensor board = x.narrow(1, 0, 768).view({B, 12, 8, 8}).to(target_dtype);
        // Selected piece: 1 channel
        Tensor selected = x.narrow(1, 853, 64).view({B, 1, 8, 8}).to(target_dtype);
        // Valid pieces: 1 channel
        Tensor valid_pieces = x.narrow(1, 917, 64).view({B, 1, 8, 8}).to(target_dtype);
        // Valid dests: 1 channel
        Tensor valid_dests = x.narrow(1, 981, 64).view({B, 1, 8, 8}).to(target_dtype);
        // Geometric planes: 4 channels (pre-computed buffer)
        Tensor geo = square_geo_planes.to(target_dtype).expand({B, -1, -1, -1});

        // Cat spatial input: (B, 19, 8, 8)
        Tensor spatial = torch::cat({board, selected, valid_pieces, valid_dests, geo}, 1);

        // 2. CNN pipeline: 1×1 embed → 1×1 project → 3×3 depthwise mix + residual
        Tensor h = torch::relu(square_embed->forward(spatial));
        h = torch::relu(channel_proj->forward(h));
        h = h + torch::relu(spatial_mix->forward(h));  // residual connection
        Tensor board_features = h.flatten(1);  // (B, 64 * PROJ_DIM = 512)

        // 3. Promotions: flat binary mask (B, 32)
        Tensor promos = (x.narrow(1, 1045, 32) > 0).to(target_dtype);

        // 4. Categorical embeddings (one-hot → argmax → embedding lookup)
        Tensor side_f = side_embed->forward(x.narrow(1, 768, 2).argmax(1)).to(target_dtype);
        Tensor castle_f = castle_embed->forward(x.narrow(1, 770, 16).argmax(1)).to(target_dtype);
        Tensor ep_f = ep_embed->forward(x.narrow(1, 786, 65).argmax(1)).to(target_dtype);
        Tensor phase_f = phase_embed->forward(x.narrow(1, 851, 2).argmax(1)).to(target_dtype);

        // 5. Scalars: normalized to [0, 1]
        Tensor scalars = x.narrow(1, 1077, 5).to(target_dtype) / 255.0;

        // 6. Concatenate all features and project
        Tensor features = torch::cat({board_features, promos, side_f, castle_f, ep_f, phase_f, scalars}, 1);
        return torch::relu(proj->forward(features));
    }
};

// ChessTwo encoder: 4-layer residual CNN (256 channels) with scalar MLP
// Matches Python ChessTwo architecture for ~8.5M param chess network
// Obs layout identical to ChessEncoder (1082 bytes per player)
class ChessTwoEncoder : public Encoder {
    public:
        static constexpr int CNN_CH = 256;
        static constexpr int EMBED_DIM = 32;
        static constexpr int SPATIAL_IN = 16; // 12 pieces + selected + valid_pieces + valid_dests + promos_padded

        nn::Conv2d conv1{nullptr};  // (16, 256, 3x3)
        nn::Conv2d conv2{nullptr};  // (256, 256, 3x3) - residual block
        nn::Conv2d conv3{nullptr};  // (256, 256, 3x3) - residual block
        nn::Conv2d conv4{nullptr};  // (256, hidden, 3x3)

        nn::Embedding side_embed{nullptr};
        nn::Embedding castle_embed{nullptr};
        nn::Embedding ep_embed{nullptr};
        nn::Embedding phase_embed{nullptr};

        nn::Linear scalar_fc1{nullptr};
        nn::Linear scalar_fc2{nullptr};

        nn::Linear proj{nullptr};

        int input, hidden;

    ChessTwoEncoder(int input, int hidden) : input(input), hidden(hidden) {
        conv1 = register_module("conv1", nn::Conv2d(nn::Conv2dOptions(SPATIAL_IN, CNN_CH, 3).padding(1)));
        nn::init::orthogonal_(conv1->weight, std::sqrt(2.0));
        nn::init::constant_(conv1->bias, 0.0);

        conv2 = register_module("conv2", nn::Conv2d(nn::Conv2dOptions(CNN_CH, CNN_CH, 3).padding(1)));
        nn::init::orthogonal_(conv2->weight, std::sqrt(2.0));
        nn::init::constant_(conv2->bias, 0.0);

        conv3 = register_module("conv3", nn::Conv2d(nn::Conv2dOptions(CNN_CH, CNN_CH, 3).padding(1)));
        nn::init::orthogonal_(conv3->weight, std::sqrt(2.0));
        nn::init::constant_(conv3->bias, 0.0);

        conv4 = register_module("conv4", nn::Conv2d(nn::Conv2dOptions(CNN_CH, hidden, 3).padding(1)));
        nn::init::orthogonal_(conv4->weight, std::sqrt(2.0));
        nn::init::constant_(conv4->bias, 0.0);

        side_embed = register_module("side_embed", nn::Embedding(2, EMBED_DIM));
        castle_embed = register_module("castle_embed", nn::Embedding(16, EMBED_DIM));
        ep_embed = register_module("ep_embed", nn::Embedding(65, EMBED_DIM));
        phase_embed = register_module("phase_embed", nn::Embedding(2, EMBED_DIM));

        scalar_fc1 = register_module("scalar_fc1", nn::Linear(5, hidden));
        nn::init::orthogonal_(scalar_fc1->weight, std::sqrt(2.0));
        nn::init::constant_(scalar_fc1->bias, 0.0);
        scalar_fc2 = register_module("scalar_fc2", nn::Linear(hidden, hidden));
        nn::init::orthogonal_(scalar_fc2->weight, std::sqrt(2.0));
        nn::init::constant_(scalar_fc2->bias, 0.0);

        // CNN flat: hidden*64, embeddings: 4*32=128, scalar: hidden
        int cnn_flat = hidden * 64;
        int total = cnn_flat + 4 * EMBED_DIM + hidden;
        proj = register_module("proj", nn::Linear(total, hidden));
        nn::init::orthogonal_(proj->weight, std::sqrt(2.0));
        nn::init::constant_(proj->bias, 0.0);
    }

    Tensor forward(Tensor x) override {
        int64_t B = x.size(0);
        auto target_dtype = conv1->weight.dtype();

        // Spatial: 12 piece planes + selected + valid_pieces + valid_dests + promos_padded = 16ch
        Tensor board = x.narrow(1, 0, 768).view({B, 12, 8, 8}).to(target_dtype);
        Tensor selected = x.narrow(1, 853, 64).view({B, 1, 8, 8}).to(target_dtype);
        Tensor valid_pieces = x.narrow(1, 917, 64).view({B, 1, 8, 8}).to(target_dtype);
        Tensor valid_dests = x.narrow(1, 981, 64).view({B, 1, 8, 8}).to(target_dtype);
        // Pad promotions from (B,32) -> (B,1,4,8) -> (B,1,8,8)
        Tensor promos_raw = x.narrow(1, 1045, 32).view({B, 1, 4, 8}).to(target_dtype);
        Tensor promos_padded = torch::zeros({B, 1, 8, 8}, torch::TensorOptions().dtype(target_dtype).device(x.device()));
        promos_padded.narrow(2, 0, 4) = promos_raw;

        Tensor spatial = torch::cat({board, selected, valid_pieces, valid_dests, promos_padded}, 1);

        // 4-layer residual CNN
        Tensor h = torch::relu(conv1->forward(spatial));
        Tensor residual = h;
        h = torch::relu(conv2->forward(h));
        h = conv3->forward(h);
        h = torch::relu(h + residual);
        h = torch::relu(conv4->forward(h));
        Tensor cnn_features = h.flatten(1);  // (B, hidden*64)

        // Categorical embeddings
        Tensor side_f = side_embed->forward(x.narrow(1, 768, 2).argmax(1)).to(target_dtype);
        Tensor castle_f = castle_embed->forward(x.narrow(1, 770, 16).argmax(1)).to(target_dtype);
        Tensor ep_f = ep_embed->forward(x.narrow(1, 786, 65).argmax(1)).to(target_dtype);
        Tensor phase_f = phase_embed->forward(x.narrow(1, 851, 2).argmax(1)).to(target_dtype);

        // Scalar MLP
        Tensor scalars = x.narrow(1, 1077, 5).to(target_dtype) / 255.0;
        Tensor scalar_out = torch::relu(scalar_fc1->forward(scalars));
        scalar_out = torch::relu(scalar_fc2->forward(scalar_out));

        // Concatenate and project
        Tensor features = torch::cat({cnn_features, side_f, castle_f, ep_f, phase_f, scalar_out}, 1);
        return torch::relu(proj->forward(features));
    }
};

// Create policy with env-specific encoder/decoder
Policy* create_policy(const std::string& env_name, int input_size, int hidden_size,
        int decoder_output_size, int num_layers, int act_n, bool is_continuous,
        bool kernels, int chess_encoder) {
    shared_ptr<Encoder> enc;
    shared_ptr<Decoder> dec;
    if (env_name == "puffer_snake") {
        enc = std::make_shared<SnakeEncoder>(input_size, hidden_size, 8);
        dec = std::make_shared<DefaultDecoder>(hidden_size, decoder_output_size, is_continuous);
    } else if (env_name == "falsepuffer_g2048") {
        //TODO: This encoder is worse (hence commented with falsepuffer)
        enc = std::make_shared<SimpleG2048Encoder>(input_size, hidden_size);
        dec = std::make_shared<DefaultDecoder>(hidden_size, decoder_output_size, is_continuous);
    } else if (env_name == "puffer_nmmo3") {
        enc = std::make_shared<NMMO3Encoder>(input_size, hidden_size);
        dec = std::make_shared<NMMO3Decoder>(hidden_size, decoder_output_size);
    } else if (env_name == "puffer_drive") {
        enc = std::make_shared<DriveEncoder>(input_size, hidden_size);
        dec = std::make_shared<DefaultDecoder>(hidden_size, decoder_output_size, is_continuous);
    } else if (env_name == "puffer_chess") {
        // chess_encoder: 1=ChessEncoder (fast), 2=ChessTwoEncoder (stronger, default)
        if (chess_encoder == 1) {
            enc = std::make_shared<ChessEncoder>(input_size, hidden_size);
        } else {
            enc = std::make_shared<ChessTwoEncoder>(input_size, hidden_size);
        }
        dec = std::make_shared<DefaultDecoder>(hidden_size, decoder_output_size, is_continuous);
    } else {
        enc = std::make_shared<DefaultEncoder>(input_size, hidden_size);
        dec = std::make_shared<DefaultDecoder>(hidden_size, decoder_output_size, is_continuous);
    }
    auto rnn = std::make_shared<MinGRU>(hidden_size, num_layers, kernels);
    return new Policy(enc, dec, rnn, input_size, act_n, hidden_size);
}
