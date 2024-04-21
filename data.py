import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import chess

SYMBOL_TO_INDEX = { symbol: index for index, symbol in enumerate([None, "R", "N", "B", "Q", "K", "P", "r", "n", "b", "q", "k", "p"]) }

ROOT_DIR = Path(__file__).parent

def parse_eval(eval):
    if isinstance(eval, str):
        return torch.tensor(1 if eval[1] == '+' else -1)
    return torch.tanh(torch.tensor(eval * 0.01))

def FEN_to_tiles(fen: str):
    board = chess.Board(fen)
    tiles = []
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        tiles.append(SYMBOL_TO_INDEX[piece.symbol()] if piece else 0)
    return tiles

class Incremental(Dataset):
    def __init__(self, path, num_samples):
        self.path = path
        self.len = num_samples

    def __getitem__(self, index):
        sample = pd.read_csv(self.path, skiprows = index + 1, nrows = 1, names=["FEN", "eval"])
        return torch.tensor(FEN_to_tiles(sample["FEN"][0])), parse_eval(sample["eval"][0]).float()

    def __len__(self):
        return self.len


# train_size = 12954834
# test_size = 1000273
# train_dataset = Incremental(ROOT_DIR / "dataset" / "chessData.csv", num_samples = train_size)
# # test_dataset = Incremental(ROOT_DIR / "dataset" / "random_evals.csv", num_samples = test_size)
# train_loader = DataLoader(train_dataset, batch_size = 32, shuffle = False)
# # test_loader = DataLoader(test_dataset, batch_size = 32, shuffle = False)

# for i, (states, evals) in enumerate(train_loader):
#     sample = next(iter(train_loader))
#     print(sample[0].shape, sample[1].shape)
#     if i > 10: break
