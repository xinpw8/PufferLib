#include "flat_chess_env.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <cstring>

// Utility: split whitespace preserving tokens
static std::vector<std::string> tokenize(const std::string& line) {
    std::istringstream iss(line);
    std::vector<std::string> toks;
    std::string tok;
    while (iss >> tok) {
        toks.push_back(tok);
    }
    return toks;
}

static bool is_move_number(const std::string& tok) {
    size_t pos = tok.find('.');
    if (pos == std::string::npos) return false;
    return std::all_of(tok.begin(), tok.begin() + pos, ::isdigit);
}

static bool is_result_token(const std::string& tok) {
    return tok == "1-0" || tok == "0-1" || tok == "1/2-1/2" || tok == "*";
}

class CerrSilencer {
    std::streambuf* old_ = nullptr;
    std::ostringstream sink_;
public:
    CerrSilencer() { old_ = std::cerr.rdbuf(sink_.rdbuf()); }
    ~CerrSilencer() { std::cerr.rdbuf(old_); }
};

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <moves.txt>\n";
        return 1;
    }
    std::ifstream fin(argv[1]);
    if (!fin) { std::cerr << "File open failed\n"; return 1; }
    std::ostringstream oss; oss << fin.rdbuf();
    auto tokens = tokenize(oss.str());

    CChess env{}; allocate(&env);
    int ply = 0; bool error = false;
    for (const auto& tok : tokens) {
        if (is_move_number(tok) || is_result_token(tok)) continue;
        ++ply;
        CerrSilencer sil;
        auto maybe = env.board.ParseSANMove(tok);
        if (!maybe) { --ply; continue; }
        auto mv = *maybe;
        env.actions[0] = ((mv.from.y*8 + mv.from.x) << 6) | (mv.to.y*8 + mv.to.x);
        c_step(&env);
        if (env.terminals[0]) { std::cout << "[info] terminal after ply "<<ply<<" "<<tok<<"\n"; break; }
    }
    std::cout << env.board.DebugString();
    free_allocated(&env);
    return error?2:0;
} 