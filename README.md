# CNNChess
A convolutional neural network trained on Stockfish to evaluate chess boards, implemented into an engine.

## How does it work?
I'll be brief, but I'd recommend you check out Sebastian Lague's chess engine [two-part series](https://www.youtube.com/watch?v=U4ogK0MIzqk&t=11s) on YouTube.

Chess engines function—on a very basic level—similarly to humans. They (1) determine what moves they can make, (2) they evaluate how good these moves are, and (3) they decide which move to play.

The first step, determining the moves, is done by a move generator. I did not code this, as chess rules are incredibly complicated and nuanced.

The second step, evaluating the moves, is done by my CNN engine. Given a chess board, the CNN will output a float; the greater the float is, the better the position is for white. The model was trained to guess how good a board is for white and then check its guess with Stockfish's guess.

The final step, deciding on a move, is done by the minimax algorithm. It's normally paired with alpha-beta pruning to reduce the number of moves to evaluate. It is not as simple as just picking the move with the highest score!

## Setup
The setup for the chess engine is split up into two segments: training the CNN and compiling the C++ chess engine.

> [!WARNING]
> This engine can only be installed on Linux and *maybe* WSL! This is because Libtorch doesn't like being installed on Windows. This chess engine also requires you to have an Nvidia GPU.

### 1. Training the CNN model
1. Clone the repository and change directories to `./cnn/data`.
2. Download the Lichess database from [here](https://database.lichess.org/#evals) and move it to `./cnn/data`.
3. Create chunks of data.
   - I recommend 269 chunks of 10,000 boards each.
   - To do this, enter `python3 data.py 269 10000`.
4. Change directories to `./cnn`.
5. Begin training the model as well as possible without having it diverge.
   - I recommend starting at 269 chunks of 10,000 boards each and reducing the number of chunks from there.
   - To do this, enter `python3 torch_eval.py 269 10000`.
     
![image of training data](/cnn/references/referenceTrain.png)

Here is an example of the model training data. Ideally, you should train the model until the model diverges at around 170 chunks.

![image of testing data](/cnn/references/referenceTest.png)

This testing data of the model is very similar to the training data.

You should end with testing data that only shows a slight, downward trend in the loss. Place the resulting `cppEvalModel.pt` model file into a new directory called `./engine/build`.

### 2. Creating the chess engine

1. Download the CUDA 12.1, cxx11 ABI version of Libtorch [here](https://pytorch.org/) and extract it to `./engine`.
2. Download files `./src/thc.h` and `./src/thc.cpp` from the [Triplehappy Chess](https://github.com/billforsternz/thc-chess-library/tree/master) repository.
   - Create new directory `./engine/thc-chess-library` and place them there.
3. Change directories into `./engine/build`.
4. Enter `cmake -DCMAKE_PREFIX_PATH=/absolute/path/to/extracted_libtorch_folder -DTorch_DIR=/absolute/path/to/extracted_libtorch_folder/libtorch/share/cmake/Torch ..`
5. Enter `cmake --build . --config Release`.

After this, you should have a chess engine executable at `./engine/build/cnnChess`.

## Running the engine in console (not recommended)

The engine follows a very bare-bone implementation of the Universal Chess Interface ([UCI](https://wbec-ridderkerk.nl/html/UCIProtocol.html)).
While in the build directory, enter `./cnnChess`. It takes no arguments and is a mini-console.

1. To load the CNN, enter `isready` in the mini-console.
2. To set the position, enter one of the following:
   - `position startpos` will set the engine's position to the beginning chess position.
   - `position fen [fen]` will set the engine's position to the FEN string entered.
   - `position fen [fen] moves [...moves]` will set the engine's position to the FEN string entered, plus the moves after the argument "moves".
3. To compute the best move from the engine's position, enter the following:
   - `go wtime [wtime] btime [btime] winc [winc] binc [binc]`
   - "wtime" is the time left on white's clock
   - "btime" is the time on black's clock
   - "winc" is how much white's clock goes up when their turn ends
   - "binc" is how much black's clock goes up when their turn ends

## Running the engine in UCIvsGUI

[UCIvsGUI](https://github.com/EvanA4/UCIvsGUI) is my own GUI for playing against UCI engines.

To run it, simply grab the engine binary `./engine/build/cnnChess` and the CNN model `./engine/build/cppEvalModel.py` and place them in the same directory as `UCIvsGUI/play.py`. To run the GUI while working in the same directory, enter `python3 play.py -e ./cnnChess`. Make sure you have `chess`, `cairosvg`, and `pygame` installed with pip in whatever Python environment you're using.

For more usage instructions for UCIvsGUI, check out the repository.


