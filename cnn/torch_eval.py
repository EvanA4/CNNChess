import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import sys
import pathlib
import os
import matplotlib.pyplot as plt

# Learning rate data
# .1    : 9
# .05   : 10
# .01   : 9.76
# .001  : 9.44
# .0001 : 9 <- much less noise

# Batch size at LR = .0001
# 64  : 9
# 128 : 8.5
# 256 : 8.4

# Hidden shape at LR = .0001 and BS = 256
# 64  : 8.4 <- much faster
# 32  : 8.82
# 128 : 8.4 

# Epochs at at LR = .0001, BS = 256, and HS = 64
# 1   : 8.4
# 10  : 8
# 50  : 6.6
# 100 : 5.54

# Learning speed: 1000 chunks takes 49 minutes at 10 epochs
# ~5 min before model diverges in training

BATCH_SIZE = 256
EPOCHS = 1
HIDDEN_SHAPE = 64
LEARNING_RATE = .0001
DATA_SPLIT = .9
KERNEL_SIZE = 3


# custom dataset code credit to https://github.com/adam-abed-abud/chess_engine_CNN/blob/master/pytorch_CNN_chess_engine.ipynb
class chess_dataset(Dataset):   
    def __init__(self, boards, scores):
        self.boards = boards
        self.scores = scores
        
    def __getitem__(self, index):
        return self.boards[index], self.scores[index]
        
    def __len__ (self):
        return len(self.boards)
    
    def __getXshape__(self):
        return self.boards.size()

    def __getYshape__(self):
        return self.scores.size()


class chessCNN_M(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels = 6,
                      out_channels = HIDDEN_SHAPE,
                      kernel_size = KERNEL_SIZE,
                      stride = 1,
                      padding = 0)
        self.bn1 = nn.BatchNorm2d(HIDDEN_SHAPE)
        self.conv2 = nn.Conv2d(in_channels = HIDDEN_SHAPE,
                      out_channels = HIDDEN_SHAPE * 2,
                      kernel_size = KERNEL_SIZE,
                      stride = 1,
                      padding = 0)
        self.bn2 = nn.BatchNorm2d(HIDDEN_SHAPE * 2)
        self.linear1 = nn.Linear(in_features = HIDDEN_SHAPE * 2 * (8 - 2 * (KERNEL_SIZE - 1)) * (8 - 2 * (KERNEL_SIZE - 1)),
                                out_features = HIDDEN_SHAPE * 2 * (8 - 2 * (KERNEL_SIZE - 1)) * (8 - 2 * (KERNEL_SIZE - 1)))
        self.linear2 = nn.Linear(in_features = HIDDEN_SHAPE * 2 * (8 - 2 * (KERNEL_SIZE - 1)) * (8 - 2 * (KERNEL_SIZE - 1)),
                                out_features = HIDDEN_SHAPE * 2 * (8 - 2 * (KERNEL_SIZE - 1)) * (8 - 2 * (KERNEL_SIZE - 1)))
        self.linear3 = nn.Linear(in_features = HIDDEN_SHAPE * 2 * (8 - 2 * (KERNEL_SIZE - 1)) * (8 - 2 * (KERNEL_SIZE - 1)),
                                out_features = HIDDEN_SHAPE * 2 * (8 - 2 * (KERNEL_SIZE - 1)) * (8 - 2 * (KERNEL_SIZE - 1)))
        self.linear4 = nn.Linear(in_features = HIDDEN_SHAPE * 2 * (8 - 2 * (KERNEL_SIZE - 1)) * (8 - 2 * (KERNEL_SIZE - 1)),
                                out_features = HIDDEN_SHAPE * 2 * (8 - 2 * (KERNEL_SIZE - 1)) * (8 - 2 * (KERNEL_SIZE - 1)))
        self.linear5 = nn.Linear(in_features = HIDDEN_SHAPE * 2 * (8 - 2 * (KERNEL_SIZE - 1)) * (8 - 2 * (KERNEL_SIZE - 1)),
                                out_features = 1)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.selu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = torch.selu(x)
        x = torch.flatten(x, 1, -1)
        x = self.linear1(x)
        x = torch.selu(x)
        x = self.linear2(x)
        x = torch.selu(x)
        x = self.linear3(x)
        x = torch.selu(x)
        x = self.linear4(x)
        x = torch.selu(x)
        x = self.linear5(x)
        return torch.squeeze(x)


if __name__ == "__main__":
    # error handle command line arguments
    if (len(sys.argv) != 3):
        print("usage: python multipletorch_eval.py [number of chunks] [chunk size]", file=sys.stderr)
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

    datapath = pathlib.Path(__file__).parent.resolve().__str__() + "/data"
    if not os.path.exists(datapath + f"/{CHUNKSIZE}chunks"):
        print("Error:", datapath + f"/{CHUNKSIZE}chunks", "does not exist, data must be generated first")

    for i in range(1, numChunks + 1):
        if not os.path.exists(datapath + f"/{CHUNKSIZE}chunks/{i}.pkl"):
            print(f"Error: file ./data/{CHUNKSIZE}chunks/{i}.pkl does not exist", file=sys.stderr)
            exit()

    # initialize model, loss fn, and optimizer
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("\nUsing device \"" + str(device) + "\"")
    model = chessCNN_M()
    model.to(device)
    loss_fn = nn.L1Loss()
    opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    numBatches = 0
    DATA_SPLIT = int(DATA_SPLIT * 10000)
    b_vals = []
    l_vals = []
    test_vals = []
    chunk_vals = list(range(1, numChunks + 1))

    for i in range(numChunks):
        print("\n\nReading", f"./data/{CHUNKSIZE}chunks/{i + 1}.pkl")
        df = pd.read_pickle(f"./data/{CHUNKSIZE}chunks/{i + 1}.pkl")

        r_train = chess_dataset(torch.FloatTensor(df["Boards"][:DATA_SPLIT].values.tolist()), torch.FloatTensor(df["Score"][:DATA_SPLIT].values.tolist()))
        r_test = chess_dataset(torch.FloatTensor(df["Boards"][DATA_SPLIT:].values.tolist()), torch.FloatTensor(df["Score"][DATA_SPLIT:].values.tolist()))
        rtrain_load = DataLoader(dataset=r_train, batch_size=BATCH_SIZE, shuffle=True)
        rtest_load = DataLoader(dataset=r_test, batch_size=BATCH_SIZE, shuffle=True)

        # training loop
        model.train()
        total_step = len(rtrain_load)
        for epoch in range(1, EPOCHS + 1):
            epoch_loss = 0
            for batch, (x_batch, y_batch) in enumerate(rtrain_load):
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                y_pred = model(x_batch)
                loss = loss_fn(y_pred, y_batch) 

                opt.zero_grad()
                loss.backward()
                opt.step()
                epoch_loss += loss.item()
                            
                if (batch % 1000 == 0):
                    print('Epoch [{:3d}/{}], Loss: {:4d}'.format(epoch, EPOCHS, int(loss.item()) // 64))
                    
                b_val = batch + (epoch * total_step) + numBatches * i
                if (b_val % 10 == 0):
                    b_vals.append(b_val)
                    l_vals.append(int(loss.item()) // 64)
        
        # testing loop
        print("Testing...")
        numTestBatches = 0
        avgTestLoss = 0
        with torch.inference_mode():
            for batch, (x_batch, y_batch) in enumerate(rtest_load):
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                y_pred = model(x_batch)
                loss = loss_fn(y_pred, y_batch) 

                avgTestLoss += loss.item() // 64
                numTestBatches += 1
        
        test_vals.append(avgTestLoss / numTestBatches)

        if i == 0:
            numBatches = b_vals[-1]
    
    # save model
    print("\n\nSaving model and displaying results...\n")
    torch.save(model, f"eval_model_{EPOCHS}epochs")
    scriptmodel = torch.jit.script(model)
    scriptmodel.save("cppEvalModel.pt")

    # plot model training data, then testing data
    plt.plot(b_vals, l_vals)
    plt.title("Train Loss")
    plt.xlabel("Batch")
    plt.ylabel("Loss")
    plt.savefig(f"{EPOCHS}epochtrain.png")
    plt.show()

    plt.plot(chunk_vals, test_vals)
    plt.title("Test Loss")
    plt.xlabel("Chunk Number")
    plt.ylabel("Average Loss")
    plt.savefig(f"{EPOCHS}epochtest.png")
    plt.show()
