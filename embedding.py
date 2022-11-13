from torch import nn, device, cuda
import torch

# device
dv = device("cuda:0" if cuda.is_available() else "cpu")


class positional_encoding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim) -> None:
        super().__init__()
        pos = torch.arange(0, num_embeddings)
        i = torch.arange(0, embedding_dim)
        
        i = (2 * (i//2)) / embedding_dim
        i = 1 / torch.pow(10000, i)
        self.mat = torch.outer(pos, i).to(dv)

        self.mat[:,::2] = torch.sin(self.mat[:,::2])
        self.mat[:,1::2] = torch.cos(self.mat[:,1::2])

    def forward(self, x):
        return self.mat[:x.size(1),:] + x

class transformer_embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, padding_idx) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings+100, embedding_dim, padding_idx=padding_idx)
        self.positional_encoding = positional_encoding(512, embedding_dim)

    def forward(self, x):
        embedded_x = self.embedding(x)
        return self.positional_encoding(embedded_x)
