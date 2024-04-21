
import chess
import torch
from torch import nn

class Evaluate(nn.Module):
    def __init__(self, embedding_dim = 32, nhead = 4, num_layers = 2, dim_feedforward = 64, dropout = 0.1):
        super(Evaluate, self).__init__()

        self.embedding = nn.Embedding(8 * 8, embedding_dim - 2) # leave room for position embedding
        encoder_layer = nn.TransformerEncoderLayer(embedding_dim, nhead, dim_feedforward, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.decoder = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward * embedding_dim, 1),
            nn.Tanh()
        )

    def forward(self, tiles):
        batch_size = tiles.shape[0]

        tile_embeddings = self.embedding(tiles)
        
        pos_encodings = torch.tensor([[pos // 8, pos % 8] for pos in range(64)]).repeat(batch_size, 1, 1).to(tile_embeddings.device)
        tile_embeddings = torch.cat([tile_embeddings, pos_encodings], dim=2)
        
        tile_embeddings = self.transformer_encoder(tile_embeddings)
        return self.decoder(tile_embeddings.view(batch_size, -1))
