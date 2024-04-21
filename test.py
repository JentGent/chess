import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader
from data import Incremental
import random
import chess
import model

ROOT_DIR = Path(__file__).parent

def parse_eval(eval):
    return float(eval[1:] if eval[0] == "#" else eval)

train_size = 12954834
data = pd.read_csv(Path(__file__).parent / "dataset" / "chessData.csv", names = ["FEN", "eval"], skiprows = 1, nrows = 10000)
for i in range(10):
    print(data["FEN"][i + 100].split(" ")[1:])
# print(data["eval"].apply(parse_eval).mean())
print(f"{0:03b}")


train_size = 12954834
test_size = 1000273
train_dataset = Incremental(ROOT_DIR / "dataset" / "chessData.csv", num_samples = train_size)
# test_dataset = Incremental(ROOT_DIR / "dataset" / "random_evals.csv", num_samples = test_size)
train_loader = DataLoader(train_dataset, batch_size = 32, shuffle = False)
# test_loader = DataLoader(test_dataset, batch_size = 32, shuffle = False)

for i, (states, evals) in enumerate(train_loader):
    sample = next(iter(train_loader))
    print(sample[0].shape, sample[1].shape)
    if i > 10: break

state, eval = next(iter(train_loader))
print(model.Evaluate().forward(state))

