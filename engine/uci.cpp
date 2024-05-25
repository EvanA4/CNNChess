#include <iostream>
#include <fstream>
#include <chrono>
#include <string>
#include <vector>
#include <unordered_map>
#include <torch/script.h>
#include "uci.hpp"

// move generator, move push/pop, and mate detection by https://github.com/billforsternz/thc-chess-library
// demo code at https://github.com/billforsternz/thc-chess-library/blob/master/src/demo.cpp
#include "thc-chess-library/thc.h"


// determines time bot can search for best move
void UCI::get_search_time(int wtime, int btime, int winc, int binc) {
    // function credit to Sebastian Lague:
    // https://github.com/SebLague/Chess-Coding-Adventure/blob/Chess-V2-UCI/Chess-Coding-Adventure/src/Bot.cs

    int myTimeRemainingMs = isWhite ? wtime : btime;
    int myIncrementMs = isWhite ? winc : binc;
    // Get a fraction of remaining time to use for current move
    double thinkTimeMs = myTimeRemainingMs / 40.0;
    // Add increment
    if (myTimeRemainingMs > myIncrementMs * 2) {
        thinkTimeMs += myIncrementMs * 0.8;
    }

    double minThinkTime = myTimeRemainingMs * 0.25;
    if (50 < minThinkTime) minThinkTime = 50;
    
    searchTime = ceil(minThinkTime > thinkTimeMs ? minThinkTime : thinkTimeMs);
}


// returns move as a string and its eval
float UCI::minimax(thc::ChessRules &cr, int depth, bool isWhite) {
    // if we're past time, stop
    int duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - searchStart).count();
    if (duration > searchTime) {
        return OUT_OF_TIME;
    }

    if (depth == 0) {
        return eval();
    }

    if (isWhite) {
        float maxEval = -__FLT_MAX__;
        
        // generate moves
        std::vector<thc::Move> moves;
        std::vector<bool> check;
        std::vector<bool> mate;
        std::vector<bool> stalemate;
        cr.GenLegalMoveList(  moves, check, mate, stalemate );
        unsigned int len = moves.size();

        // for each child of position
        for( unsigned int i=0; i<len; i++ ) {
            float currentEval;

            if (mate[i]) {
                currentEval = 10000.F;

            } else if (stalemate[i]) {
                currentEval = .0F;

            } else {
                cr.PushMove(moves[i]);
                currentEval = minimax(cr, depth - 1, !isWhite);
                cr.PopMove(moves[i]);
            }

            // std::cout << depth << " " << moves[i].TerseOut() << ": " << currentEval << std::endl;
            if (currentEval == OUT_OF_TIME) {
                return OUT_OF_TIME;
            }

            if (currentEval > maxEval) {
                maxEval = currentEval;

                if (depth == searchDepth) {
                    bestMove = moves[i];
                }
            }
        }

        return maxEval;
    }

    else {
        float maxEval = __FLT_MAX__;
        
        // generate moves
        std::vector<thc::Move> moves;
        std::vector<bool> check;
        std::vector<bool> mate;
        std::vector<bool> stalemate;
        cr.GenLegalMoveList(  moves, check, mate, stalemate );
        unsigned int len = moves.size();

        // for each child of position
        for( unsigned int i=0; i<len; i++ ) {
            float currentEval;

            if (mate[i]) {
                currentEval = -10000.F;

            } else if (stalemate[i]) {
                currentEval = .0F;

            } else {
                cr.PushMove(moves[i]);
                currentEval = minimax(cr, depth - 1, !isWhite);
                cr.PopMove(moves[i]);
            }

            // std::cout << depth << " " << moves[i].TerseOut() << ": " << currentEval << std::endl;
            if (currentEval == OUT_OF_TIME) {
                return OUT_OF_TIME;
            }

            if (currentEval < maxEval) {
                maxEval = currentEval;

                if (depth == searchDepth) {
                    bestMove = moves[i];
                }
            }
        }

        return maxEval;
    }

    return .0F;
}


// determines best move given a bunch of arguments
void UCI::go(std::vector<std::string> &args) {
    searchStart = std::chrono::high_resolution_clock::now();

    if (args.size() != 9) {
        std::cerr << "usage: go wtime [wtime] btime [btime] winc [winc] binc [binc]" << std::endl;
        return;
    }

    // read args and get search time
    int wtime = std::stoi(args[2]);
    int btime = std::stoi(args[4]);
    int winc = std::stoi(args[6]);
    int binc = std::stoi(args[8]);
    get_search_time(wtime, btime, winc, binc);
    // std::cout << searchTime << std::endl;

    // iterative deepening DFS
    searchDepth = 1;
    thc::Move backup;

    while (true) {
        // std::cout << "Beginning search of depth " << searchDepth << std::endl;
        if (minimax(cr, searchDepth, isWhite) == OUT_OF_TIME) break;
        // std::cout << "current move: " << bestMove.TerseOut() << std::endl;
        ++searchDepth;
        backup = bestMove;
    }

    // finally, print the best move
    std::cout << "bestmove " << backup.TerseOut() << std::endl;
}


// creates board position for "go" command
void UCI::position(std::vector<std::string> &args) {
    // if startpos, then quickly return
    if (args.size() == 2 && args[1] == "startpos") {
        if (args.size() == 2) {
            cr.Forsyth("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
            isWhite = true;
            return;
        }
    }

    // basic error handling
    if (args.size() == 1 || args[1] != "fen" || (args.size() > 8 && args[8] != "moves")) {
        std::cerr << "Error: invalid arguments" << std::endl;
        std::cerr << "usage 1: position startpos" << std::endl;
        std::cerr << "usage 2: position fen <FEN> moves <move1> ..." << std::endl;
        return;
    }

    // find initial FEN to play moves from
    std::string startFEN = args[2];
    std::vector<std::string>::iterator it = args.begin() + 3;
    while (it != args.end() && *it != "moves") {
        startFEN += " " + *it;
        ++it;
    }

    // initialize chess board
    cr.Forsyth(startFEN.c_str());
    isWhite = cr.WhiteToPlay();

    // play moves, if any
    if (it != args.end()) {
        ++it;
        while (it != args.end()) {

            thc::Move mv;
            mv.TerseIn(&cr, (*it).c_str());
            cr.PlayMove(mv);

            ++it;
        }
    }
}


// evaluates current position with the CNN
double UCI::eval() {
    if (!modelLoaded) {
        std::cerr << "Error: model has not been loaded" << std::endl;
        return -1.0;
    }

    // torch::Tensor boards = torch::full({1, 6, 8, 8}, .0F, torch::TensorOptions().device(torch::kCUDA, 0));
    torch::Tensor boards = torch::full({1, 6, 8, 8}, .0F);

    // first convert FEN string to boards
    std::string board = cr.ForsythPublish();

    int ctr = 0;
    for (int i = 0; i < (int) board.length(); ++i) {
        if (board[i] == ' ') break;
        if (board[i] == '/') continue;

        if (pieceIdxs.find(board[i]) == pieceIdxs.end()) { // if int, skip
            ctr += board[i] - '0';
        
        } else { // else, add to appropriate board
            boards[0][pieceIdxs[board[i]] % 6][ctr / 8][ctr % 8] = 1.0F - 2.0F * (pieceIdxs[board[i]] < 5);
            ++ctr;
        }
    }

    // then evaluate
    std::vector<torch::jit::IValue> inputs;
    // inputs.push_back(boards); // for CPU
    inputs.push_back(boards.to(torch::kCUDA)); // load on GPU
    return model.forward(inputs).toTensor().item<float>();
}


// loads the given CNN file and move generator databases
void UCI::load_assets(std::string file) {
    try {
        // model = torch::jit::load(file);
        model = torch::jit::load(file, torch::kCUDA); // load on GPU

        modelLoaded = true;

    } catch (const c10::Error& e) {
        std::cerr << "Error: failed to load model" << std::endl;
    }
}