import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
import torch.optim as optim


class CNNLayerNorm(nn.Module):
    """Layer normalization built for cnns input"""
    def __init__(self, n_feats):
        super(CNNLayerNorm, self).__init__()
        self.layer_norm = nn.LayerNorm(n_feats)

    def forward(self, x):
        # x (batch, channel, feature, time)
        x = x.transpose(2, 3).contiguous() # (batch, channel, time, feature)
        x = self.layer_norm(x)
        return x.transpose(2, 3).contiguous() # (batch, channel, feature, time) 


class ResidualCNN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride, dropout, n_feats):
        super().__init__()
        self.cnn1 = nn.Conv2d(in_channels, out_channels, kernel, stride, padding=kernel//2)
        self.cnn2 = nn.Conv2d(out_channels, out_channels, kernel, stride, padding=kernel//2)
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
        return x # (batch, channel, feature, time)
        

class BidirectionalLSTM(nn.Module):
    def __init__(self, rnn_dim, hidden_size, rnn_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=rnn_dim, hidden_size=hidden_size,
            num_layers=rnn_layers, bidirectional=True, batch_first=False)
        self.layer_norm = nn.LayerNorm(rnn_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.layer_norm(x)
        x = F.gelu(x)
        x = x.permute(1, 0, 2)
        x, hidden = self.lstm(x)
        x = self.dropout(x)
        return x, hidden


class CNNEncoder(nn.Module):
    def __init__(
        self, 
        n_cnn_layers=3, 
        hidden_size=256, 
        rnn_layers=2, 
        n_feats=128, 
        stride=2, 
        dropout=0.3
    ):
        super().__init__()
        n_feats = n_feats // 2
        self.hidden_size = hidden_size

        self.cnn = nn.Conv2d(1, 32, 3, stride=stride, padding=3//2)  # cnn for extracting heirachal features
        self.rescnn_layers = nn.Sequential(*[
            ResidualCNN(32, 32, kernel=3, stride=1, dropout=dropout, n_feats=n_feats) 
            for _ in range(n_cnn_layers)
        ]) # n residual cnn layers with filter size of 32

        self.fully_connected = nn.Linear(n_feats * 32, hidden_size)
        self.rnn_layer = BidirectionalLSTM(hidden_size, hidden_size, rnn_layers, dropout)

    def forward(self, x):
        out = x.transpose(1, 2) # (batch, feature, time)
        out = out.unsqueeze(1) # (batch, channel, time, feature)
        out = self.cnn(out)
        out = self.rescnn_layers(out)
        sizes = out.size()
        out = out.view(sizes[0], sizes[1] * sizes[2], sizes[3]).transpose(1, 2) # flatten (batch, time, feature)
        out = self.fully_connected(out)
        out, hidden = self.rnn_layer(out)
        out = out[:, :, :self.hidden_size] + out[:, :, self.hidden_size:]
        return out, hidden


class Encoder(nn.Module):
    def __init__(self, n_feats=40, hidden_size=256, num_layers=2):
        super().__init__()
        self.n_feats = n_feats
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.dropout = nn.Dropout(0.1)        
        self.lstm = nn.LSTM(self.n_feats, self.hidden_size, num_layers=self.num_layers, bidirectional=True)

    def forward(self, input_tensor):
        # input_tensor = pack_padded_sequence(input_tensor, input_feature_lengths, enforce_sorted=False)
        output, hidden = self.lstm(input_tensor)
        # output = pad_packed_sequence(output)[0]
        output = output[:, :, :self.hidden_size] + output[:, :, self.hidden_size:]         
        output = self.dropout(output)
        return output, hidden


# The decoder consists of LSTM and attention mechanism. It is initialized using the hidden states of the encoder and uses the vector representation from the encoder to predict the next character, conditioned on the previous predictions. For the attention mechanism, we will use hybrid + location-aware attention, explained in more detail [here](https://proceedings.neurips.cc/paper/2015/file/1068c6e4c8051cfd4e9ea8072e3189e2-Paper.pdf). The decoder is defined in the cell below:

class Decoder(nn.Module):
    def __init__(
        self, 
        embedding_dim=128, 
        encoder_hidden_size=256, 
        attention_hidden_size=256, 
        n_class=30, 
        rnn_layers=1, 
        num_filters=128,
        dropout=0.3,
        device="cpu",
    ):        
        super().__init__()
        self.device = device
        self.embedding = nn.Embedding(n_class, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, encoder_hidden_size, num_layers=rnn_layers, bidirectional=False, batch_first=False)
        self.out = nn.Linear(encoder_hidden_size * 2, n_class)
        self.dropout = nn.Dropout(dropout)
        
        self.v = nn.Parameter(torch.FloatTensor(1, encoder_hidden_size).uniform_(-0.1, 0.1))
        self.b = nn.Parameter(torch.FloatTensor(encoder_hidden_size).uniform_(-0.1, 0.1))
        self.W_1 = torch.nn.Linear(encoder_hidden_size, attention_hidden_size, bias=False)
        self.W_2 = torch.nn.Linear(encoder_hidden_size, attention_hidden_size, bias=False)
        self.W_3 = nn.Linear(num_filters, attention_hidden_size, bias=False)
        self.conv = nn.Conv1d(in_channels=1, out_channels=num_filters, kernel_size=3, padding=1)
    
    def hybrid_attention_score(self, encoder_output, decoder_output, conv_feat):
        out = torch.tanh(self.W_1(decoder_output) + self.W_2(encoder_output) + self.W_3(conv_feat) + self.b)
        v = self.v.repeat(encoder_output.data.shape[1], 1).unsqueeze(1)
        out = out.permute(1, 0, 2)
        v = v.permute(0, 2, 1)
        scores = out.bmm(v)
        return scores
        
    def forward(self, input_tensor, decoder_hidden, encoder_output, attn_weights):
        embedding = self.embedding(input_tensor)
        embedding = embedding.permute(1, 0, 2)

        decoder_output, decoder_hidden = self.rnn(embedding, decoder_hidden)
        try:
            conv_feat = self.conv(attn_weights).permute(0, 2, 1)
        except:
            random_tensor = torch.rand(encoder_output.size(1), 1, encoder_output.size(0)).to(self.device)
            conv_feat = self.conv(F.softmax(random_tensor, dim=-1)).to(self.device).permute(0, 2, 1)
 
        conv_feat = conv_feat.permute(1, 0, 2)
        scores = self.hybrid_attention_score(encoder_output, decoder_output, conv_feat)
        scores = scores.permute(1, 0, 2)
        attn_weights = F.softmax(scores, dim=0)
 
        context = torch.bmm(attn_weights.permute(1, 2, 0), encoder_output.permute(1, 0, 2))
        context = context.permute(1, 0, 2)
        output = torch.cat((context, decoder_output), -1)

        output = self.out(output[0])
        output = self.dropout(output)
        output = F.log_softmax(output, 1)
        return output, decoder_hidden, attn_weights
    

# encoder_layers = 5
# decoder_layers = 1

# encoder_hidden_size = 150
# attention_hidden_size = 150

# embedding_dim_chars = 100
# num_filters = 100

# encoder_lr = 0.0005
# decoder_lr = 0.0005

# num_epochs = 10
# MAX_LENGTH = 800
# skip_training = True


# # initialize the Encoder
# encoder = Encoder(features_train[0].size(1), encoder_hidden_size, encoder_layers).to(device)
# encoder_optimizer = optim.Adam(encoder.parameters(), lr=encoder_lr)

# # initialize the Decoder
# decoder = Decoder(embedding_dim_chars, encoder_hidden_size, attention_hidden_size, len(char2idx)+1, decoder_layers, encoder_layers, num_filters, batch_size, device).to(device)
# decoder_optimizer = optim.Adam(decoder.parameters(), lr=decoder_lr)

