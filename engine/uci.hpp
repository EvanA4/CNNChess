#include <string>
#include <vector>
#include <chrono>
#include <torch/script.h>
// #include "chess-library/include/chess.hpp"
#include "thc-chess-library/thc.h"


class UCI {
    std::chrono::_V2::system_clock::time_point searchStart;
    int searchDepth, searchTime;
    thc::Move bestMove;
    bool isWhite;
    const float OUT_OF_TIME = * (float *) "time"; // lol funny out of time constant

    bool modelLoaded = false;
    torch::jit::script::Module model;
    std::unordered_map<char, int> pieceIdxs = {
        {'p', 0},
        {'r', 1},
        {'n', 2},
        {'b', 3},
        {'q', 4},
        {'k', 5},
        {'P', 6},
        {'R', 7},
        {'N', 8},
        {'B', 9},
        {'Q', 10},
        {'K', 11}
    };
    thc::ChessRules cr;

    // void push_move(std::vector<std::string> &fen, int &halfMoves, int &fullMoves, std::string &move);
    void get_search_time(int wtime, int btime, int winc, int binc);
    float minimax(thc::ChessRules &cr, int depth, bool isWhite);

    public:
        void go(std::vector<std::string> &args);
        void position(std::vector<std::string> &args);
        double eval();
        void load_assets(std::string file);
};