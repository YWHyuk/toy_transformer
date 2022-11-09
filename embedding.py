from torch import nn
import torch

def positional_encoding(num_embeddings, embedding_dim):
    pos = torch.arange(0, num_embeddings)
    i = torch.arange(0, embedding_dim)
    
    i = (2 * (i//2)) / embedding_dim
    i = 1 / torch.pow(10000, i)
    mat = torch.outer(pos, i)

    mat[:,::2] = torch.sin(mat[:,::2])
    mat[:,1::2] = torch.cos(mat[:,1::2])
    return mat

class transformer_embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.positional_encoding = positional_encoding(512, embedding_dim)

    def forward(self, x):
        embedded_x = self.embedding(x)
        encoded_x = self.positional_encoding[:embedded_x.size(1),:] + embedded_x
        return encoded_x
