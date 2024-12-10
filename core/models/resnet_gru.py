import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Multinomial, Categorical

import math
import numpy as np


class CNNLayerNorm(nn.Module):
    """Layer normalization built for cnns input"""

    def __init__(self, n_feats):
        super(CNNLayerNorm, self).__init__()
        self.layer_norm = nn.LayerNorm(n_feats)

    def forward(self, x):
        # x (batch, channel, feature, time)
        x = x.transpose(2, 3).contiguous()  # (batch, channel, time, feature)
        x = self.layer_norm(x)
        # (batch, channel, feature, time)
        return x.transpose(2, 3).contiguous()


class ResidualCNN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride, dropout, n_feats):
        super(ResidualCNN, self).__init__()

        self.cnn1 = nn.Conv2d(in_channels, out_channels,
                              kernel, stride, padding=kernel//2)
        self.cnn2 = nn.Conv2d(out_channels, out_channels,
                              kernel, stride, padding=kernel//2)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.layer_norm1 = CNNLayerNorm(n_feats)
        self.layer_norm2 = CNNLayerNorm(n_feats)

    def forward(self, x):
        residual = x  # (batch, channel, feature, time)
        x = self.layer_norm1(x)
        x = F.gelu(x)
        x = self.dropout1(x)
        x = self.cnn1(x)
        x = self.layer_norm2(x)
        x = F.gelu(x)
        x = self.dropout2(x)
        x = self.cnn2(x)
        x += residual
        return x  # (batch, channel, feature, time)


class BidirectionalGRU(nn.Module):

    def __init__(self, rnn_dim, hidden_size, dropout, batch_first):
        super(BidirectionalGRU, self).__init__()

        self.BiGRU = nn.GRU(
            input_size=rnn_dim, hidden_size=hidden_size,
            num_layers=1, batch_first=batch_first, bidirectional=True)
        self.layer_norm = nn.LayerNorm(rnn_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.layer_norm(x)
        x = F.gelu(x)
        x, _ = self.BiGRU(x)
        x = self.dropout(x)
        return x


class SpeechRecognitionModel(nn.Module):
    def __init__(self, n_cnn_layers=3, n_rnn_layers=5, rnn_dim=256, n_class=30, n_feats=128, stride=2, dropout=0.1, lm_scorer=None, acoustics_weight=0.7):
        super(SpeechRecognitionModel, self).__init__()
        n_feats = n_feats//2
        # cnn for extracting heirachal features
        self.cnn = nn.Conv2d(1, 32, 3, stride=stride, padding=3//2)

        # n residual cnn layers with filter size of 32
        self.rescnn_layers = nn.Sequential(*[
            ResidualCNN(32, 32, kernel=3, stride=1,
                        dropout=dropout, n_feats=n_feats)
            for _ in range(n_cnn_layers)
        ])
        self.fully_connected = nn.Linear(n_feats*32, rnn_dim)
        self.birnn_layers = nn.Sequential(*[
            BidirectionalGRU(rnn_dim=rnn_dim if i == 0 else rnn_dim*2,
                             hidden_size=rnn_dim, dropout=dropout, batch_first=i == 0)
            for i in range(n_rnn_layers)
        ])
        self.attention = nn.MultiheadAttention(
            embed_dim=rnn_dim*2, num_heads=2)
        self.classifier = nn.Sequential(
            nn.Linear(rnn_dim*2, rnn_dim),  # birnn returns rnn_dim*2
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(rnn_dim, n_class),
            nn.LogSoftmax(dim=-1)
        )
        self.lm_scorer = lm_scorer

    def forward(self, x):
        x = x.transpose(1, 2)  # (batch, feature, time)
        x = x.unsqueeze(1)  # (batch, channel, time, feature)
        x = self.cnn(x)
        x = self.rescnn_layers(x)
        sizes = x.size()
        x = x.view(sizes[0], sizes[1] * sizes[2],
                   sizes[3])  # (batch, feature, time)
        x = x.transpose(1, 2)  # (batch, time, feature)
        x = self.fully_connected(x)
        x = self.birnn_layers(x)
        x, _ = self.attention(x, x, x)
        x = self.classifier(x)
        return x

    def decode_batch(self, logits, input_length, vocab):
        pred_ids = torch.argmax(logits, dim=-1)
        decoded_texts = []
        for i in range(logits.shape[0]):
            unique_ids = pred_ids[i][-input_length[i]:]
            decoded_text = "".join([vocab[id.item()] for id in unique_ids])
            decoded_texts.append(decoded_text)
        return decoded_texts

    def greedy_decode_batch(self, logits, input_length, vocab):
        pred_ids = torch.argmax(logits, dim=-1)
        decoded_texts = []
        for i in range(logits.shape[0]):
            unique_ids = torch.unique_consecutive(
                pred_ids[i][-input_length[i]:])
            unique_ids = [id for id in unique_ids if id != 0]
            decoded_text = "".join([vocab[id.item()] for id in unique_ids])
            decoded_texts.append(decoded_text.strip())
        return decoded_texts

    def score_decode_batch(self, logits, input_length, vocab, k=100, lm_weight=0.7, include_best_acoustic=False):
        if self.lm_scorer is None or lm_weight is None or lm_weight == 0:
            return self.greedy_decode_batch(logits, input_length, vocab)

        def decode_text(topk_ids):
            unique_ids = torch.unique_consecutive(
                topk_ids[:][-input_length[i]:])
            unique_ids = [id for id in unique_ids if id != 0]

            return "".join([vocab[id.item()] for id in unique_ids])

        def do_score(topk_ids, i, text):
            prob = logits[i].gather(1, topk_ids.unsqueeze(-1)).squeeze(-1) 
            
            acoustic_score = prob[-input_length[i]:].sum().item()

            lm_score = self.lm_scorer.score(text)

            return (acoustic_score + lm_weight * lm_score, text)

        decoded_texts = []

        if include_best_acoustic:
            best_predict = torch.argmax(logits, dim=-1)

        dist = Categorical(logits=logits)
        samples = dist.sample((k,))

        for i in range(logits.shape[0]):
            seen = set()
            sequences = []

            # Make sure to sample the top acoustic prediction
            if include_best_acoustic:
                t = decode_text(best_predict[i])
                seen.add(t)
                sequences.append(do_score(best_predict[i], i, t))

            for j in range(k):
                # Sample from top predictions
                text = decode_text(samples[j, i])
                if text in seen:
                    continue

                seen.add(text)
                seq = do_score(samples[j, i], i, text)
                sequences.append(seq)

            # Sort based on score
            sequences.sort(reverse=True)
            sentence = sequences[0][1]
            decoded_texts.append(sentence.strip())

        debug = False

        if debug:
            greedy = self.greedy_decode_batch(logits, input_length, vocab)
            if greedy != decoded_texts:
                for i in range(len(greedy)):
                    if greedy[i] != decoded_texts[i]:
                        print("Greedy ", greedy[i])
                        print("Decoded", decoded_texts[i])
            else:
                pass
                # print("No change after lm score")
        return decoded_texts



