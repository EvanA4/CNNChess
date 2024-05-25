#include <iostream>
#include <vector>
#include <torch/script.h>
#include "uci.hpp"
// #include "chess-library/include/chess.hpp"
#include "thc-chess-library/thc.h"


// splits command from miniterminal into separate arguments
void parse_args(std::string &cmd, std::vector<std::string> &args) {
    args.clear();

    int idx = 0;
    int len = 0;
    for (std::string::iterator it = cmd.begin(); it != cmd.end(); ++it) {
        if (*it == ' ') {
            args.push_back(cmd.substr(idx, len));
            idx += len + 1;
            len = -1;
        }

        ++len;
    }

    if (len) {
        args.push_back(cmd.substr(idx, len));
    }
}


int main() {
    c10::InferenceMode guard;
    std::vector<std::string> args;
    std::string cmd;
    UCI engine;

    while (true) {
        std::getline(std::cin, cmd);
        parse_args(cmd, args);

        if (args[0] == "uci") {
            std::cout << "uciok" << std::endl;

        } else if (args[0] == "isready") {
            engine.load_assets("/home/evana/Documents/myPrograms/projects/chessEval/eval_cnn.pt");
            std::cout << "readyok" << std::endl;

        } else if (args[0] == "position") {
            engine.position(args);

        } else if (args[0] == "go") {
            engine.go(args);

        } else if (args[0] == "eval") {
            std::cout << engine.eval() << std::endl;

        } else if (args[0] == "quit") {
            exit(0);
        }
    }
}

/*
cmake --build . --config Release
go wtime 300000 btime 300000 winc 2000 binc 2000
position fen rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 moves b1c3
*/