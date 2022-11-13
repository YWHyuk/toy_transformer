from torch import nn, matmul, transpose, cat
from math import sqrt

class singlehead_attention(nn.Module):
    def __init__(self, d_model, d_k, d_v):
        super().__init__()
        self.d_k = d_k
        self.d_v = d_v

        self.weight_q = nn.Linear(d_model, d_k)
        self.weight_k = nn.Linear(d_model, d_k)
        self.weight_v = nn.Linear(d_model, d_v)
        self.soft_max = nn.Softmax(dim=1)
        
    def forward(self, q, k, v, mask):
        # Projecting query, key, value
        projected_q = self.weight_q(q)
        projected_k = self.weight_k(k)
        projected_v = self.weight_v(v)
        
        # Calculate attention
        score = matmul(projected_q, transpose(projected_k, -2, -1)) / sqrt(self.d_v)
        if mask is not None:
            # Mask score
            score = score.masked_fill(mask, float("-Inf"))
        weight = self.soft_max(score)

        attention = matmul(weight, projected_v)
        return attention

class multihead_attention(nn.Module):
    def __init__(self, d_model, d_k, d_v, h):
        super().__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.h = h

        self.weight_o = nn.Linear(h*d_v, d_model)
        self.attentions = nn.ModuleList([singlehead_attention(d_model, d_k, d_v) for i in range(h)])

    def forward(self, q, k, v, mask=None):
        attention_list = [attention(q, k ,v, mask) for attention in self.attentions]
        attention = cat(attention_list, dim=-1)
        result = self.weight_o(attention)
        return result

if __name__ == "__main__":
    d_model = 10
    d_k = 8
    d_v = 6
    h = 16
    tokens = 250

    import torch

    m = multihead_attention(d_model, d_k, d_v, h)
    q = torch.randn((tokens, d_model), dtype=torch.float32)
    k = torch.randn((tokens, d_model), dtype=torch.float32)
    v = torch.randn((tokens, d_model), dtype=torch.float32)
    m(q, k, v)
    