import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader
import data
import random
import chess
import model

ROOT_DIR = Path(__file__).parent

train_size = 12954834
d = pd.read_csv(Path(__file__).parent / "dataset" / "chessData.csv", names = ["FEN", "eval"], skiprows = 1, nrows = 10000)
for i in range(10):
    print(d["FEN"][i + 100].split(" ")[1:])
    print(data.parse_eval(d["eval"][i + 100]))
print(d["eval"].apply(data.parse_eval).mean())
print(f"{0:03b}")


train_size = 12954834
test_size = 1000273
train_dataset = data.DatasetFromPath(ROOT_DIR / "dataset" / "chessData.csv")
print(len(train_dataset))
test_dataset = data.DatasetFromPath(ROOT_DIR / "dataset" / "random_evals.csv")
train_loader = DataLoader(train_dataset, batch_size = 32, shuffle = False)
test_loader = DataLoader(test_dataset, batch_size = 32, shuffle = False)

for i, (states, evals) in enumerate(train_loader):
    sample = next(iter(train_loader))
    print(sample[0].shape, sample[1].shape)
    if i > 10: break

state, eval = next(iter(train_loader))
print(model.Evaluate().forward(state))

print(model.encode_position(3, 4))
