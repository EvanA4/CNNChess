import json
import pandas as pd
import sys
import pathlib
import os
import numpy as np


# DATASET = lichess_db_eval.json
# find dataset at https://database.lichess.org/#evals
# recommended: at most 269 chunks at size 10000 (model likely diverges before that)
# time for recommended: ~21 min


PIECES_DICT = {
    'p': 0,
    'r': 1,
    'n': 2,
    'b': 3,
    'q': 4,
    'k': 5,
    'P': 6,
    'R': 7,
    'N': 8,
    'B': 9,
    'Q': 10,
    'K': 11,
}


# adapted from adam-abed-abud's chess_engine_CNN at https://github.com/adam-abed-abud/chess_engine_CNN/tree/master
def fen_to_NPYbitboards(fen: str):
    pieces = [[[0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0]],
              [[0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0]],
              [[0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0]],
              [[0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0]],
              [[0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0]],
              [[0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0]]]

    fen = fen.split(" ")[0]
    fen = fen.split("/")
    ctr = 0
    for row in fen:
        for char in row:
            if (char.isdigit()):
                ctr += int(char)
                continue
            else:
                if (PIECES_DICT[char] < 6):
                    pieces[PIECES_DICT[char]][ctr >> 3][ctr & 7] = -1
                else:
                    pieces[PIECES_DICT[char] - 6][ctr >> 3][ctr & 7] = 1
            ctr += 1

    return pieces


if __name__ == "__main__":
    # error handle command line arguments
    if len(sys.argv) != 3:
        print("usage: python data.py [number of chunks] [chunk size]", file=sys.stderr)
        exit()

    numChunks = 0
    try:
        numChunks = int(sys.argv[1])
    except:
        print("Error: invalid number of chunks", file=sys.stderr)
        exit()

    CHUNKSIZE = 0
    try:
        CHUNKSIZE = int(sys.argv[2])
    except:
        print("Error: invalid chunk size", file=sys.stderr)
        exit()

    if CHUNKSIZE > 20999266 or CHUNKSIZE < 1:
        print(f"Error: chunk size must be an integer in range [1, 20999266]", file=sys.stderr)
        exit()

    if numChunks > 20999266 // CHUNKSIZE or numChunks < 1:
        print(f"Error: for chunk size {CHUNKSIZE}, number of chunks must be an integer in range [1, {20999266 // CHUNKSIZE}]", file=sys.stderr)
        exit()

    datapath = pathlib.Path(__file__).parent.resolve().__str__() # absolute directory of this file
    if not os.path.exists(datapath + f"/{CHUNKSIZE}chunks"):
        os.mkdir(datapath + f"/{CHUNKSIZE}chunks")

    # begin reading data
    f = open((datapath + "/lichess_db_eval.json").replace("\\", "/"))
    ctr = 0
    df = pd.DataFrame(columns=['Boards', 'Score'])

    while ctr < numChunks * CHUNKSIZE:
        if (os.path.exists(datapath + f"/{CHUNKSIZE}chunks/" + str(ctr // CHUNKSIZE + 1) + '.pkl')):
            print("Skipping", str(ctr // CHUNKSIZE + 1) + '.pkl')
            ctr += CHUNKSIZE
            continue

        jsonString = f.readline()
        if jsonString == "": break
        board_data = json.loads(jsonString)

        score = 0
        if "cp" in board_data["evals"][0]["pvs"][0]:
            score = board_data["evals"][0]["pvs"][0]["cp"]
        else:
            if board_data["evals"][0]["pvs"][0]["mate"] > 0:
                score = 10000
            else:
                score = -10000

        boards = fen_to_NPYbitboards(board_data["fen"])

        df.loc[ctr] = [boards, score]

        if (ctr % CHUNKSIZE == CHUNKSIZE - 1):
            filename = datapath + f"/{CHUNKSIZE}chunks/" + str(ctr // CHUNKSIZE + 1) + ".pkl"
            df.to_pickle(filename)
            df = pd.DataFrame(columns=['Boards', 'Score'])
            print(f"Generated chunk file {filename}.")
        ctr += 1

    print("\n\nData fully generated!")