from transformer import encoder_block, decoder_block, positional_encoding
from torch import nn, bmm
from embedding import transformer_embedding

class tranformer(nn.Module):
    def __init__(self, N, d_model, d_ff, d_k, d_v, h, source_vocab, target_vocab, padding_idx, p=0.1) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h
        self.N = N
        self.p = p
        self.padding_idx = padding_idx

        self.input_embedding = transformer_embedding(source_vocab, d_model)
        self.output_embedding = transformer_embedding(target_vocab, d_model)
        self.encoder = [encoder_block(d_model, d_ff, d_k, d_v, h) for _ in range(N)]
        self.decoder = [decoder_block(d_model, d_ff, d_k, d_v, h) for _ in range(N)]
        self.linear = nn.Linear(d_model, target_vocab)
        self.softmax = nn.Softmax()

    def forward(self, source, target):
        # Generate Mask
        enc_mask = self.padding_mask(source, source)
        dec_mask = self.padding_mask(target, target)
        encdec_mask = self.padding_mask(source, target)

        # embedding phase
        encoder_input = self.input_embedding(source)
        decoder_output = self.output_embedding(target)

        for enc in self.encoder:
            encoder_input = enc(encoder_input, enc_mask)
        
        for dec in self.decoder:
            decoder_output = dec(decoder_output, encoder_input, encoder_input, dec_mask, encdec_mask)

        # Linear + Softmax phase
        temp = self.linear(decoder_output)
        return self.softmax(temp)

    def padding_mask(self, source, target):
        target = target.ne(self.padding_idx).int()
        source = source.ne(self.padding_idx).int()
        result = bmm(target.unsqueeze(2), source.unsqueeze(1)).bool()
        return result
