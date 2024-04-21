import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import chess
import model

SYMBOL_TO_INDEX = { symbol: index for index, symbol in enumerate([None, "R", "N", "B", "Q", "K", "P", "r", "n", "b", "q", "k", "p"]) }
NUM_SYMBOLS = len(SYMBOL_TO_INDEX)

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
    
    fields = fen.split(" ")
    active_color = [1] if fields[1] == "w" else [-1]
    castling = [
        1 if "K" in fields[2] else -1,
        1 if "Q" in fields[2] else -1,
        1 if "k" in fields[2] else -1,
        1 if "q" in fields[2] else -1
    ]
    if fields[3] == "-": en_passant = [0, 0, 0, 0, 0, 0]
    else: en_passant = model.encode_position(8 - int(fields[3][1]), "abcdefgh".index(fields[3][0]))
    return en_passant + castling + active_color + tiles

class Incremental(Dataset):
    def __init__(self, path, num_samples):
        self.path = path
        self.len = num_samples

    def __getitem__(self, index):
        sample = pd.read_csv(self.path, skiprows = index + 1, nrows = 1, names = ["FEN", "eval"])
        return torch.tensor(FEN_to_tiles(sample["FEN"][0])), parse_eval(sample["eval"][0]).float()

    def __len__(self):
        return self.len

