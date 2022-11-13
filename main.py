import time, math
from model import tranformer
from dataloader import dataloader
from torch import optim, nn, no_grad, save, device, cuda
from epoch import epoch_time
from blue import get_bleu

# device
dv = device("cuda:0" if cuda.is_available() else "cpu")

# optimizer parameter setting
batch_size = 512
factor = 0.9
patience = 5
warmup = 100
epoch = 100
clip = 1.0
weight_decay = 5e-4
adam_eps = 5e-9
init_lr = 0.1

# Dataloader
dl = dataloader()
source_size = len(dl.source_vocab)
target_size = len(dl.target_vocab)
padding_idx = dl.source_vocab['<sos>']

def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.kaiming_uniform(m.weight.data)

def train(model, train_data, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    train_data_len = (train_data.dataset.num_lines + train_data.batch_size -1) // train_data.batch_size
    for i, batch in enumerate(train_data):
        src = batch[0].to(dv)
        trg = batch[1].to(dv)

        optimizer.zero_grad()
        output = model(src, trg[:, :-1])
        output_reshape = output.contiguous().view(-1, output.shape[-1])
        trg = trg[:, 1:].contiguous().view(-1)

        loss = criterion(output_reshape, trg)
        loss.backward()
        #nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        epoch_loss += loss.item()
        print('step :', round((i / train_data_len) * 100, 2), '% , loss :', loss.item())
    return epoch_loss / train_data_len


def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    batch_bleu = []
    eval_data_len = (iterator.dataset.num_lines + iterator.batch_size -1) // iterator.batch_size
    with no_grad():
        for i, batch in enumerate(iterator):
            src = batch[0].to(dv)
            trg = batch[1].to(dv)

            output = model(src, trg[:, :-1])
            output_reshape = output.contiguous().view(-1, output.shape[-1])
            trg = trg[:, 1:].contiguous().view(-1)

            loss = criterion(output_reshape, trg)
            epoch_loss += loss.item()

            total_bleu = []
            for j in range(len(batch[1])):
                try:
                    trg_words = dl.idx_to_word(batch[1][j], source=False)
                    output_words = output[j].max(dim=1)[1]
                    output_words = dl.idx_to_word(output_words, source=False)
                    bleu = get_bleu(hypotheses=output_words.split(), reference=trg_words.split())
                    total_bleu.append(bleu)
                except:
                    pass

            total_bleu = sum(total_bleu) / len(total_bleu)
            batch_bleu.append(total_bleu)

    batch_bleu = sum(batch_bleu) / len(batch_bleu)
    return epoch_loss / eval_data_len, batch_bleu


def run(model, optimizer, scheduler, criterion, dl, total_epoch, best_loss):
    train_losses, test_losses, bleus = [], [], []
    for step in range(total_epoch):
        train_data, valid_data, _ = dl.datasets(batch=batch_size)
        start_time = time.time()
        train_loss = train(model, train_data, optimizer, criterion, clip)
        valid_loss, bleu = evaluate(model, valid_data, criterion)
        end_time = time.time()

        if step > warmup:
            scheduler.step(valid_loss)

        train_losses.append(train_loss)
        test_losses.append(valid_loss)
        bleus.append(bleu)
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_loss:
            best_loss = valid_loss
            save(model.state_dict(), 'saved/model-{0}.pt'.format(valid_loss))

        f = open('result/train_loss.txt', 'w')
        f.write(str(train_losses))
        f.close()

        f = open('result/bleu.txt', 'w')
        f.write(str(bleus))
        f.close()

        f = open('result/test_loss.txt', 'w')
        f.write(str(test_losses))
        f.close()

        print(f'Epoch: {step + 1} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.6f} | Train PPL: {math.exp(train_loss):7.6f}')
        print(f'\tVal Loss: {valid_loss:.6f} |  Val PPL: {math.exp(valid_loss):7.6f}')
        print(f'\tBLEU Score: {bleu:.6f}')

if __name__ == "__main__":
    t = tranformer(
        N=6, d_model=512, d_ff=2048, 
        d_k=64, d_v=64, h=8, 
        source_vocab=source_size,
        target_vocab=target_size,
        padding_idx=padding_idx
    ).to(dv)
    t.apply(initialize_weights)

    optimizer = optim.Adam(params=t.parameters(),
                            lr=init_lr,
                            weight_decay=weight_decay,
                            eps=adam_eps)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                 verbose=True,
                                                 factor=factor,
                                                 patience=patience)

    criterion = nn.CrossEntropyLoss(ignore_index=padding_idx)

    run(t, optimizer, scheduler, criterion, dl, total_epoch=epoch, best_loss=float("inf"))
