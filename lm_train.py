# %%
import math
import IPython
import matplotlib.pyplot as plt
from tqdm import tqdm, trange

import torch
import torchaudio
from torch.utils.data import DataLoader

from core.models.CTCDecoder import GreedyCTCDecoder
from core.geo_dataloader import GeoDataset, collate_fn
from core.models.RNN import LmRNN, T_UNKOWN, T_EOF, T_START
import typing


# %%
# Load dataset and dataloader
mel_spec_transform = torchaudio.transforms.MelSpectrogram(
    n_fft=256, n_mels=20, window_fn=torch.hann_window)
dataset = GeoDataset(split="train", transform_fn=mel_spec_transform)
dataloader = DataLoader(dataset, batch_size=1,
                        shuffle=True, collate_fn=collate_fn)
print(dataset)


# %%


def get_batch(data, seq_len, num_batches, idx):
    src = data[:, idx:idx+seq_len]
    target = data[:, idx+1:idx+seq_len+1]
    return src, target


def evaluate(model: LmRNN, data, criterion, batch_size, seq_len, device):

    epoch_loss = 0
    model.eval()
    num_batches = data.shape[-1]
    data = data[:, :num_batches - (num_batches - 1) % seq_len]
    num_batches = data.shape[-1]

    hidden = model.init_hidden(batch_size, device)

    with torch.no_grad():
        for idx in trange(0, num_batches - 1, seq_len, desc='Evaluating: ', leave=False, position=1):
            hidden = model.detach_hidden(hidden)
            src, target = get_batch(data, seq_len, num_batches, idx)
            src, target = src.to(device), target.to(device)
            batch_size = src.shape[0]

            prediction, hidden = model(src, hidden)
            prediction = prediction.reshape(batch_size * seq_len, -1)
            target = target.reshape(-1)

            loss = criterion(prediction, target)
            epoch_loss += loss.item() * seq_len
    return epoch_loss / num_batches


def train(model: LmRNN, data: torch.Tensor, optimizer, criterion, batch_size: int, seq_len: int, clip, device):

    epoch_loss = 0
    model.train()
    # drop all batches that are not a multiple of seq_len
    num_batches = data.shape[-1]
    data = data[:, :num_batches - (num_batches - 1) % seq_len]
    num_batches = data.shape[-1]

    hidden = model.init_hidden(batch_size, device)

    # The last batch can't be a src
    for idx in trange(0, num_batches - 1, seq_len, desc='Training: ', leave=False, position=1):
        optimizer.zero_grad()
        hidden = model.detach_hidden(hidden)

        src, target = get_batch(data, seq_len, num_batches, idx)
        src, target = src.to(device), target.to(device)
        batch_size = src.shape[0]
        prediction, hidden = model(src, hidden)

        prediction = prediction.reshape(batch_size * seq_len, -1)
        target = target.reshape(-1)
        loss = criterion(prediction, target)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item() * seq_len
    return epoch_loss / num_batches


# %%
# Build for entire dataset

def tokenize(document):
    tokens = []
    tokens.append(T_START)
    tokens.extend(document.split())
    tokens.append(T_EOF)
    return {
        "text": document,
        "tokens": tokens
    }


train_documents = []
print("Building training set...")
for features, transcript_ids, transcript in tqdm(GeoDataset(split="train", transform_fn=mel_spec_transform)):
    train_documents.append(tokenize(transcript))


def build_vocab(documents):
    vocab = {}
    # Add built in tokens
    vocab[T_START] = len(vocab)
    vocab[T_EOF] = len(vocab)
    vocab[T_UNKOWN] = len(vocab)

    for doc in documents:
        for token in doc["tokens"]:
            vocab.setdefault(token, len(vocab))

    return vocab


# %%
vocabulary = build_vocab(train_documents)

validation_documents = []
print("Building validation set...")
for features, transcript_ids, transcript in tqdm(GeoDataset(split="dev", transform_fn=mel_spec_transform)):
    validation_documents.append(tokenize(transcript))


# %%
vocab_size = len(vocabulary)
embedding_dim = 1024             # 400 in the paper
hidden_dim = 1024                # 1150 in the paper
num_layers = 2                   # 3 in the paper
dropout_rate = 0.65
tie_weights = True
lr = 1e-3                        # They used 30 and a different optimizer
batch_size = 3

model = LmRNN(
    vocab=vocabulary,
    embedding_dim=embedding_dim, 
    hidden_dim=hidden_dim, 
    num_layers=num_layers, 
    dropout_rate=dropout_rate, 
    tie_weights=tie_weights)
model.init_weights()

# model:LmRNN = torch.load("checkpoints/lm/run-1/model-epoch-12.pt")


def get_data(documents, vocab: dict, batch_size):
    data = []
    UNKNWON_VAL = vocab[T_UNKOWN]

    for document in documents:
        # vocab.get can fail, fallback to vocab[T_UNKOWN]
        mapped_doc = [vocab.get(token, UNKNWON_VAL)
                      for token in document["tokens"]]
        data.extend(mapped_doc)

    data = torch.LongTensor(data)
    num_batches = data.shape[0] // batch_size
    data = data[:num_batches * batch_size]
    data = data.view(batch_size, num_batches)
    return data


train_data = get_data(train_documents, vocabulary, batch_size)
valid_data = get_data(validation_documents, vocabulary, batch_size)


optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = torch.nn.CrossEntropyLoss()

n_epochs = 50
seq_len = 50
clip = 0.25

save_model = True

lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, factor=0.5, patience=0)

best_valid_loss = float('inf')
for epoch in range(n_epochs):
    print(f"Start epoch {epoch}")

    train_loss = train(model, train_data, optimizer, criterion,
                       batch_size, seq_len, clip, "cpu")
    valid_loss = evaluate(model, valid_data, criterion,
                          batch_size, seq_len, "cpu")

    lr_scheduler.step(valid_loss)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss

    print(f"i: {epoch}, Training loss {train_loss}")
    print(f"i: {epoch}, Validation loss {valid_loss}")
    if save_model:
        torch.save(model, f"checkpoints/lm/run-1/model-epoch-{epoch+1}.pt")


# %%
