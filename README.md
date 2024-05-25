# CNNChess
A convolutional neural network trained on Stockfish to evaluate chess boards, implemented into an engine.

## How does it work?

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
     
![image of training data](/cnn/referenceTrain.png)

Here is an example of the model training data. Ideally, you should train the model until the model diverges at around 170 chunks.

![image of testing data](/cnn/referenceTest.png)

This testing data of the model is very similar to the training data.

You should end with testing data that only shows a slight, downward trend in the loss.

### 2. Creating the chess engine
