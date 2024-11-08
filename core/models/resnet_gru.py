import torch
from torch import nn


class SimpleEncoder(nn.Module):
    def __init__(self, n_features, n_classes, hidden_size=256, num_rnn_layers=3):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(n_features, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.gru = nn.GRU(hidden_size, hidden_size=hidden_size, num_layers=num_rnn_layers, bidirectional=True, batch_first=True)
        self.gru_norm = nn.Sequential(
            nn.LayerNorm(hidden_size * 2),
            nn.ReLU(),
        )
        self.fc_out = nn.Sequential(
            nn.Linear(hidden_size * 2, n_classes),
            nn.LogSoftmax() # For CTC Loss
        )

    def forward(self, x):
        out = self.fc(x)
        out, _ = self.gru(out)
        out = self.gru_norm(out)
        out = self.fc_out(out)
        return out
    
    def decode_batch(self, logits, vocab):
        pred_ids = torch.argmax(logits, dim=-1)
        decoded_texts = []  
        for i in range(logits.shape[0]):
            unique_ids = torch.unique_consecutive(pred_ids[i])
            unique_ids = [id for id in unique_ids if id != 0]
            decoded_text = "".join([vocab[id.item()] for id in unique_ids])
            decoded_texts.append(decoded_text)
        return decoded_texts


class Encoder(nn.Module):
    def __init__(self, n_features, n_classes, hidden_size=256, num_rnn_layers=3):
        super().__init__()

        self.resnet = nn.Sequential(
            ResBlock(n_features, n_features, kernel_size=3, stride=1),
            ResBlock(n_features, n_features * 2, kernel_size=3, stride=2),
            ResBlock(n_features * 2, n_features * 2, kernel_size=3, stride=1),
        )
        self.fc = nn.Sequential(
            nn.Linear(n_features * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.gru = nn.GRU(hidden_size, hidden_size=hidden_size, num_layers=num_rnn_layers, bidirectional=True, batch_first=True)
        self.gru_norm = nn.Sequential(
            nn.LayerNorm(hidden_size * 2),
            nn.ReLU(),
        )
        self.fc_out = nn.Sequential(
            nn.Linear(hidden_size * 2, n_classes),
            nn.LogSoftmax() # For CTC Loss
        )

    def forward(self, x):
        out = x.transpose(1, 2)
        out = self.resnet(out)
        out = out.transpose(1, 2)
        out = self.fc(out)
        out, _ = self.gru(out)
        out = self.gru_norm(out)
        out = self.fc_out(out)
        return out
    
    def decode_batch(self, logits, vocab):
        pred_ids = torch.argmax(logits, dim=-1)
        decoded_texts = []  
        for i in range(logits.shape[0]):
            unique_ids = torch.unique_consecutive(pred_ids[i])
            unique_ids = [id for id in unique_ids if id != 0]
            decoded_text = "".join([vocab[id.item()] for id in unique_ids])
            decoded_texts.append(decoded_text)
        return decoded_texts


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, ):
        pass


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=kernel_size // 2),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )
        self.residual = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        residual = self.residual(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + residual
        return out