#include "flat_chess_env.h"

#define RAYLIB_STATIC
#include "raylib.h"

#include <vector>
#include <cstdlib>
#include <ctime>
#include <string>
#include <iostream>
#include <array>
#include <map>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <regex>

typedef ::Color RLColor; // raylib Color
static constexpr RLColor PUFF_RED   = {187,0,0,255};
static constexpr RLColor PUFF_CYAN  = {0,187,187,255};
static constexpr RLColor PUFF_WHITE = {241,241,241,255};
static constexpr RLColor PUFF_BG    = {6,24,24,255};
static constexpr RLColor PUFF_LINES = {50,50,50,255};

// Map OpenSpiel piece to Unicode character for drawing
static char32_t PieceToUnicode(const open_spiel::chess::Piece &p) {
    using open_spiel::chess::Color;
    using open_spiel::chess::PieceType;
    if (p.type == PieceType::kEmpty) return 0;
    switch (p.type) {
        case PieceType::kKing:   return p.color == Color::kWhite ? 0x2654 : 0x265A;
        case PieceType::kQueen:  return p.color == Color::kWhite ? 0x2655 : 0x265B;
        case PieceType::kRook:   return p.color == Color::kWhite ? 0x2656 : 0x265C;
        case PieceType::kBishop: return p.color == Color::kWhite ? 0x2657 : 0x265D;
        case PieceType::kKnight: return p.color == Color::kWhite ? 0x2658 : 0x265E;
        case PieceType::kPawn:   return p.color == Color::kWhite ? 0x2659 : 0x265F;
        default: return 0;
    }
}

static const char* PieceLetter(const open_spiel::chess::Piece &p) {
    using open_spiel::chess::PieceType;
    using open_spiel::chess::Color;
    if (p.type == PieceType::kEmpty) return " ";
    static const char* white = "KQRBNP";
    static const char* black = "kqrbnp";
    int idx = static_cast<int>(p.type) - 1; // kKing=1
    return p.color == Color::kWhite ? std::string(1, white[idx]).c_str() : std::string(1, black[idx]).c_str();
}

static void EncodeMove(const open_spiel::chess::Move &mv, CChess *env) {
    int from_idx = mv.from.y * 8 + mv.from.x;
    int to_idx   = mv.to.y   * 8 + mv.to.x;
    env->actions[0] = (from_idx << 6) | to_idx;
}

static open_spiel::chess::Move *FindMove(const std::vector<open_spiel::chess::Move> &moves,
                                        open_spiel::chess_common::Square from,
                                        open_spiel::chess_common::Square to) {
    for (const auto &m : moves) {
        if (m.from == from && m.to == to) return const_cast<open_spiel::chess::Move*>(&m);
    }
    return nullptr;
}

// Helper: convert Unicode codepoint to UTF-8 string
static std::string Utf8FromCodepoint(char32_t cp) {
    char buf[5] = {0};
    int len = 0;
    if (cp <= 0x7F) {
        buf[0] = static_cast<char>(cp);
        len = 1;
    } else if (cp <= 0x7FF) {
        buf[0] = static_cast<char>(0xC0 | ((cp >> 6) & 0x1F));
        buf[1] = static_cast<char>(0x80 | (cp & 0x3F));
        len = 2;
    } else if (cp <= 0xFFFF) {
        buf[0] = static_cast<char>(0xE0 | ((cp >> 12) & 0x0F));
        buf[1] = static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        buf[2] = static_cast<char>(0x80 | (cp & 0x3F));
        len = 3;
    } else {
        buf[0] = static_cast<char>(0xF0 | ((cp >> 18) & 0x07));
        buf[1] = static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
        buf[2] = static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        buf[3] = static_cast<char>(0x80 | (cp & 0x3F));
        len = 4;
    }
    return std::string(buf, buf + len);
}

// ----------------------------------------------------------------------------
// Sprite-sheet loading (skipped when external sprites are provided)
// ----------------------------------------------------------------------------
#ifndef RAYLIB_CHESS_EXTERNAL_SPRITES
static Texture2D gSpriteSheet{};
static std::map<std::string, Rectangle> gSpriteRects;
static std::map<std::string, Texture2D> gPieceTex; // individual textures

static void InitSprites() {
    // Try load sprite sheet first (optional)
    if (std::filesystem::exists("chess_spritesheet.png")) {
        gSpriteSheet = LoadTexture("chess_spritesheet.png");
        const int TILE_SRC = 64;
        const char *pieces[6] = {"K","Q","R","B","N","P"};
        const char *colors[2] = {"white","black"};
        for (int row = 0; row < 2; ++row) {
            for (int col = 0; col < 6; ++col) {
                std::string key = std::string(colors[row]) + "_" + pieces[col];
                Rectangle r{ (float)(col*TILE_SRC), (float)(row*TILE_SRC), (float)TILE_SRC, (float)TILE_SRC };
                gSpriteRects[key] = r;
            }
        }
    }

    // Load per-file black sprites if present
    const std::map<std::string,std::string> filenames = {
        {"black_K","bking.png"},{"black_Q","bqueen.png"},{"black_R","brook.png"},
        {"black_B","bbishop.png"},{"black_N","bknight.png"},{"black_P","bpawn.png"},
        {"white_K","wking.png"},{"white_Q","wqueen.png"},{"white_R","wrook.png"},
        {"white_B","wbishop.png"},{"white_N","wknight.png"},{"white_P","wpawn.png"}
    };
    for (auto &[key,file] : filenames) {
        if (std::filesystem::exists(file)) {
            gPieceTex[key] = LoadTexture(file.c_str());
        }
    }
}

static void UnloadSprites() {
    if (gSpriteSheet.id) UnloadTexture(gSpriteSheet);
    for (auto &[k,tex] : gPieceTex) {
        if (tex.id) UnloadTexture(tex);
    }
}

static void DrawPieceSprite(const open_spiel::chess::Piece &p, int px, int py, int tileSize) {
    using open_spiel::chess::PieceType;
    using ChessColor = open_spiel::chess::Color;
    if (p.type == PieceType::kEmpty) return;

    const char *pieceNames[7] = {"", "K","Q","R","B","N","P"};
    const char *colorStr = (p.color == ChessColor::kWhite) ? "white" : "black";
    std::string key = std::string(colorStr) + "_" + pieceNames[static_cast<int>(p.type)];
    Rectangle dst{ (float)px, (float)py, (float)tileSize, (float)tileSize };

    // prefer individual texture if available for this piece
    if (gPieceTex.count(key)) {
        Texture2D &tex = gPieceTex[key];
        Rectangle src{0,0,(float)tex.width,(float)tex.height};
        DrawTexturePro(tex, src, dst, {0,0}, 0.0f, WHITE);
        return;
    }

    // otherwise fallback to sprite sheet if loaded
    if (gSpriteSheet.id && gSpriteRects.count(key)) {
        Rectangle src = gSpriteRects[key];
        DrawTexturePro(gSpriteSheet, src, dst, {0,0}, 0.0f, WHITE);
        return;
    }
}
#endif // RAYLIB_CHESS_EXTERNAL_SPRITES

// ----------------------------------------------------------------------------
// Main
// ----------------------------------------------------------------------------
int main(int argc, char **argv) {
    std::srand(static_cast<unsigned>(std::time(nullptr)));

    // ---------------------------------------------------------------------
    // Optional PGN playback: if user supplies a path, parse it now
    // ---------------------------------------------------------------------
    std::vector<std::string> sanMoves;
    if (argc > 1) {
        std::ifstream fin(argv[1]);
        if (!fin) {
            std::cerr << "Could not open PGN/email file: " << argv[1] << std::endl;
            return 1;
        }

        std::ostringstream oss;
        oss << fin.rdbuf();
        std::string txt = oss.str();

        // Remove comments { ... }
        txt = std::regex_replace(txt, std::regex("\\{[^}]*\\}"), " ");

        // Find first occurrence of newline followed by optional spaces then "1."
        std::smatch m;
        if (std::regex_search(txt, m, std::regex("(?:\\n|^)\\s*1\\."))) {
            txt = txt.substr(m.position());
        }

        std::stringstream ss(txt);
        std::string token;
        std::regex movenum("^\\d+\\.");
        while (ss >> token) {
            if (std::regex_match(token, movenum)) continue; // move numbers
            if (token == "1-0" || token == "0-1" || token == "1/2-1/2" || token == "*") continue; // results
            sanMoves.push_back(token);
        }
    }

    // ---------------------------------------------------------------------
    CChess env{};
    allocate(&env);

    std::vector<open_spiel::chess::Move> pgnMoves;
    std::vector<bool> moveValid;
    if (!sanMoves.empty()) {
        // Convert SAN to OpenSpiel moves using scratch board and record validation
        open_spiel::chess::ChessBoard scratch = env.board; // default start
        for (auto &san : sanMoves) {
            auto maybe = scratch.ParseSANMove(san);
            if (!maybe) {
                std::cerr << "Failed to parse SAN: " << san << std::endl;
                moveValid.push_back(false);
                continue;
            }
            pgnMoves.push_back(*maybe);
            moveValid.push_back(true);
            scratch.ApplyMove(*maybe);
        }
    }

    int currentPly = 0;
    int totalTurns = (pgnMoves.size() + 1) / 2; // full moves
    open_spiel::chess::ChessBoard initialBoard = env.board;

    const int tileSize = 80;
    const int margin = 40;
    const int boardPixels = tileSize * 8;
    InitWindow(boardPixels + 2 * margin, boardPixels + 2 * margin, "PufferLib Raylib Chess");
    SetTargetFPS(60);

    // Load a Codepoint font that supports unicode chess pieces
    Font font = LoadFontEx(nullptr, 64, nullptr, 0); // default font should support

    bool squareSelected = false;
    open_spiel::chess_common::Square selected{0, 0};

    bool inputActive = false;
    std::string inputBuf;

#ifndef RAYLIB_CHESS_EXTERNAL_SPRITES
    InitSprites();
#endif

    const bool pgnMode = !pgnMoves.empty();

    while (!WindowShouldClose()) {
        // PGN navigation --------------------------------------------------
        if (pgnMode) {
            if (!inputActive && IsKeyPressed(KEY_F)) {
                inputActive = true;
                inputBuf.clear();
            }
        }
    }

    return 0;
} 