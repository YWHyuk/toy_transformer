from collections import Counter, OrderedDict
from torchtext.datasets import Multi30k
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator, vocab, Vocab
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch

def build_vocab_from_counter(counter: Counter, min_freq= 1, specials=None, special_first=True) -> Vocab:
    sorted_by_freq_tuples = sorted(counter.items(), key=lambda x: x[0])
    sorted_by_freq_tuples.sort(key=lambda x: x[1], reverse=True)
    ordered_dict = OrderedDict(sorted_by_freq_tuples)

    if specials is not None:
        if special_first:
            specials = specials[::-1]
        for symbol in specials:
            ordered_dict.update({symbol: min_freq})
            ordered_dict.move_to_end(symbol, last=not special_first)

    word_vocab = vocab(ordered_dict, min_freq=min_freq)
    return word_vocab

class dataloader():
    def __init__(self) -> None:
        self.language_pair = ("en", "de")
        self.source = get_tokenizer("spacy", language=self.language_pair[0])
        self.target = get_tokenizer("spacy", language=self.language_pair[0])
        self.build_vocab()

    def datasets(self, batch=5, tokenize=True):
        multi_datapipes = Multi30k(language_pair=self.language_pair)
        #for datapipe in multi_datapipes:
        #    datapipe.rows2columnar(self.language_pair)

        if tokenize:
            collect = self.generate_batch
        else:
            collect = None
        return [DataLoader(datapipe, batch_size=batch, collate_fn=collect) for datapipe in multi_datapipes]

    def build_vocab(self):
        # Use train dataset to build vocab
        multi_datapipe = self.datasets(batch=1, tokenize=False)

        source_counter = Counter()
        target_counter = Counter()
        for datapipe in multi_datapipe:
            for source, target in datapipe:
                source_counter.update(self.source(source[0]))
                target_counter.update(self.target(target[0]))
        self.source_vocab = build_vocab_from_counter(source_counter, specials=['<pad>', '<sos>', '<eos>'])
        self.target_vocab = build_vocab_from_counter(target_counter, specials=['<pad>', '<sos>', '<eos>'])
        self.source_vocab.set_default_index(-1)
        self.target_vocab.set_default_index(-1)
        return

    def generate_batch(self, data_batch):
        source_batch, target_batch = [], []
        for (source_item, target_item) in data_batch:
            source_item = torch.tensor([self.source_vocab[token] for token in self.source(source_item)], dtype=torch.long)
            target_item = torch.tensor([self.target_vocab[token] for token in self.target(target_item)], dtype=torch.long)
            source_batch.append(torch.cat([torch.tensor([self.source_vocab['<sos>']]), source_item, torch.tensor([self.source_vocab['<eos>']])], dim=0))
            target_batch.append(torch.cat([torch.tensor([self.target_vocab['<sos>']]), target_item, torch.tensor([self.target_vocab['<eos>']])], dim=0))
        source_batch = pad_sequence(source_batch, padding_value=self.source_vocab['<pad>'], batch_first=True)
        target_batch = pad_sequence(target_batch, padding_value=self.target_vocab['<pad>'], batch_first=True)
        return source_batch, target_batch

    def idx_to_word(self, x, source=True):
        words = []
        selected_vocab = self.source_vocab if source else self.target_vocab
        for i in x:
            word = selected_vocab.lookup_token(i)
            if '<' not in word:
                words.append(word)
        words = " ".join(words)
        return words

if __name__ == "__main__":
    k = dataloader()
    t = k.datasets()
    for i,j in t[0]:
        print(i[0], j[0])