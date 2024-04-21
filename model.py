
import chess
import torch
from torch import nn

SYMBOL_TO_INDEX = { symbol: index for index, symbol in enumerate([None, "R", "N", "B", "Q", "K", "P", "r", "n", "b", "q", "k", "p"]) }
NUM_SYMBOLS = len(SYMBOL_TO_INDEX)

def encode_position(row: int, col: int) -> list[int]:
    """
    ```
           on chess board    function input
    rows       1 to 8     =>     7 to 0
    cols       a to h     =>     0 to 7
    ```
    """
    row = f"{row:03b}"
    col = f"{col:03b}"
    return [
        1 if row[0] == "1" else -1,
        1 if row[1] == "1" else -1,
        1 if row[2] == "1" else -1,
        1 if col[0] == "1" else -1,
        1 if col[1] == "1" else -1,
        1 if col[2] == "1" else -1
    ]

class Evaluate(nn.Module):
    def __init__(self, embedding_dim = 32, nhead = 4, num_layers = 2, dim_feedforward = 64, dropout = 0.1):
        super(Evaluate, self).__init__()

        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(NUM_SYMBOLS + 1, embedding_dim)
        encoder_layer = nn.TransformerEncoderLayer(embedding_dim, nhead, dim_feedforward, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.decoder = nn.Sequential(
            nn.Linear(65 * embedding_dim, 1),
            nn.Tanh()
        )

    def forward(self, tiles):
        # tiles is (batch_size, 6 + 4 + 1 + 64 = 75)
        batch_size = tiles.shape[0]

        en_passant = tiles[:, :6]
        castling = tiles[:, 6:10]
        active_color = tiles[:, 10:11]
        tiles = tiles[:, 11:] # (batch_size, 64)

        tile_embeddings = self.embedding(tiles) # (batch_size, 64, embedding_dim)

        positional_encodings = torch.tensor([encode_position(pos // 8, pos % 8) for pos in range(64)]).repeat(batch_size, 1, 1).to(tile_embeddings.device) # (batch_size, 64, 6)
        tile_embeddings = torch.cat([tile_embeddings[:, :, :6] + positional_encodings, tile_embeddings[:, :, 6:]], dim = 2) # (batch_size, 64, embedding_dim)

        board_state = torch.tensor([[NUM_SYMBOLS] for i in range(batch_size)]).to(tiles.device) # (batch_size, 1)
        board_state = self.embedding(board_state) # (batch_size, 1, embedding_dim)
        board_state = torch.cat([board_state[:, 0, :6] + en_passant, board_state[:, 0, 6:10] + castling, board_state[:, 0, 10:11] + active_color, board_state[:, 0, 11:]], dim = 1).view(batch_size, 1, self.embedding_dim)
        tile_embeddings = torch.cat([board_state, tile_embeddings], dim = 1) # (batch_size, 65, embedding_dim)
        
        tile_embeddings = self.transformer_encoder(tile_embeddings) # (batch_size, 65, embedding_dim)
        return self.decoder(tile_embeddings.view(batch_size, -1)) # (batch_size, 1)
