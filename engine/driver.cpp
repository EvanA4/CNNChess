#include <iostream>
#include <vector>
#include <filesystem>
#include <torch/script.h>
#include "uci.hpp"
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


int main(int argc, char** argv) {

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
            std::string fileDir = argv[0];
            fileDir = fileDir.substr(0, fileDir.size() - 8) + "cppEvalModel.pt";
            engine.load_assets(fileDir);
            std::cout << "readyok" << std::endl;

        } else if (args[0] == "position") {
            engine.position(args);

        } else if (args[0] == "go") {
            std::string move = engine.go(args);
            std::cout << "bestmove " << move << std::endl;

        } else if (args[0] == "quit") {
            exit(0);
        }
    }
}

/*
To compile after making changes:
cmake --build . --config Release

Some reference commands for miniconsole:
uci
isready
go wtime 300000 btime 300000 winc 2000 binc 2000
position fen rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 moves b1c3
*/
