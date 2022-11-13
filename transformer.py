from torch import nn
from attention import multihead_attention
import torch

def positional_encoding(x):
    token, d_model = x.shape
    pos = torch.arange(0, token)
    i = torch.arange(0, d_model)
    
    i = (2 * (i//2)) / d_model
    i = 1 / torch.pow(10000, i)
    mat = torch.outer(pos, i)

    mat[:,::2] = torch.sin(mat[:,::2])
    mat[:,1::2] = torch.cos(mat[:,1::2])
    return mat + x

class feed_forward(nn.Module):
    def __init__(self, d_model, d_ff=None) -> None:
        super().__init__()
        if d_ff is None:
            d_ff = d_model*4
        self.linear1 = nn.Linear(d_model, d_ff)
        self.activation = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(d_ff, d_model)
    
    def forward(self, x):
        temp = self.linear1(x)
        temp = self.activation(temp)
        return self.linear2(temp)

class encoder_block(nn.Module):
    def __init__(self, d_model, d_ff, d_k, d_v, h, p=0.1) -> None:
        super().__init__()
        self.multihead_attention = multihead_attention(d_model, d_k, d_v, h)
        self.dropout1 = nn.Dropout(p)
        self.layer_norm1 = nn.LayerNorm([d_model])
        self.feed_forward = feed_forward(d_model, d_ff)
        self.dropout2 = nn.Dropout(p)
        self.layer_norm2 = nn.LayerNorm([d_model])

    def forward(self, x, enc_mask=None):
        # Attention phase
        attention = self.multihead_attention(x, x, x, enc_mask)
        attention = self.dropout1(attention)
        ff_input = self.layer_norm1(x + attention)
        
        # Feed forward phase
        ff_output = self.feed_forward(ff_input)
        ff_output = self.dropout2(ff_output)
        result = self.layer_norm2(ff_input + ff_output)
        return result

class decoder_block(nn.Module):
    def __init__(self, d_model, d_ff, d_k, d_v, h, p=0.1) -> None:
        super().__init__()
        self.masked_multihead_attention = multihead_attention(d_model, d_k, d_v, h)
        self.dropout1 = nn.Dropout(p)
        self.layer_norm1 = nn.LayerNorm([d_model])
        self.multihead_attention = multihead_attention(d_model, d_k, d_v, h)
        self.dropout2 = nn.Dropout(p)
        self.layer_norm2 = nn.LayerNorm([d_model])
        self.feed_forward = feed_forward(d_model, d_ff)
        self.dropout3 = nn.Dropout(p)
        self.layer_norm3 = nn.LayerNorm([d_model])

    def forward(self, x, v, k, dec_mask=None, encdec_mask=None):
        # Masked Attention phase
        attention = self.masked_multihead_attention(x, x, x, dec_mask)
        attention = self.dropout1(attention)
        attention_input = self.layer_norm1(x + attention)

        # Attention phase
        temp = self.multihead_attention(attention_input, k, v, encdec_mask)
        temp = self.dropout2(temp)
        ff_input = self.layer_norm2(temp + attention_input)

        # Feed forward phase
        ff_output = self.feed_forward(ff_input)
        ff_output = self.dropout3(ff_output)
        result = self.layer_norm2(ff_input + ff_output)
        return result

if __name__ == "__main__":
    d_model = 10
    d_k = 8
    d_v = 6
    h = 16
    batch = 32
    token = 250

    e = encoder_block(d_model, d_k, d_v, h)
    d = decoder_block(d_model, d_k, d_v, h)

    setence = torch.randn((batch, token, d_model))
    result = e(setence)
    d(setence, result, result)