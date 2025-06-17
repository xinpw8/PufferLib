// demo_chess.cc - standalone tester for the flat Chess environment (relocated)
#include "flat_chess_env.h"
#include <cstdlib>
#include <ctime>
#include <vector>
#include <iostream>
#include <string>

int main() {
    std::srand(static_cast<unsigned>(std::time(nullptr)));

    CChess env{};
    allocate(&env);

    std::cout << "PufferLib Chess demo (coordinate input e2e4, r=random, q=quit)\n";

    for (int t = 0; t < 200 && !env.terminals[0]; ++t) {
        std::vector<open_spiel::chess::Move> moves;
        env.board.GenerateLegalMoves([&](const open_spiel::chess::Move &mv) {
            moves.push_back(mv); return true;
        });
        if (moves.empty()) { std::cout << "No legal moves – game over!\n"; break; }

        std::string input; open_spiel::chess::Move chosen;
        while (true) {
            std::cout << "Your move (r/q): ";
            std::getline(std::cin, input);
            if (input == "q" || input == "quit") { return 0; }
            if (input == "r") { chosen = moves[std::rand()%moves.size()]; break; }
            if (input.size()>=4) {
                auto ff=open_spiel::chess::ParseFile(input[0]);
                auto fr=open_spiel::chess::ParseRank(input[1]);
                auto tf=open_spiel::chess::ParseFile(input[2]);
                auto tr=open_spiel::chess::ParseRank(input[3]);
                if (ff&&fr&&tf&&tr) {
                    open_spiel::chess_common::Square from{*ff,*fr};
                    open_spiel::chess_common::Square to{*tf,*tr};
                    for (auto &m:moves) if (m.from==from && m.to==to){chosen=m; goto got;}
                }
            }
            std::cout << "Illegal move.\n";
        }
        got:
        env.actions[0] = ((chosen.from.y*8+chosen.from.x)<<6)| (chosen.to.y*8+chosen.to.x);
        c_step(&env);
        c_render(&env);
    }
    free_allocated(&env);
    return 0;
} 