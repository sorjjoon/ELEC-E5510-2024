import torch
import math
import typing
# Constants for start, end and unknown tokens
T_UNKOWN = "<unk>"
T_EOF = "<eof>"
T_START = "<start>"

class LmRNN(torch.nn.Module):
    def __init__(self, vocab:typing.Dict[str, int], embedding_dim:int, hidden_dim:int, num_layers:int, dropout_rate:float, 
                tie_weights: bool):
        """Called must call init_weights, unless deserializing an instance

        """
        assert T_UNKOWN in vocab
        assert T_EOF in vocab
        assert T_START in vocab

        vocab_size = len(vocab)
        # Note, vocab is not actually used by this class, however we store it here so it's easier to use pickled models
        self.vocab = vocab
                
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim

        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)

        self.lstm = torch.nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, 
                    dropout=dropout_rate, batch_first=True)
        
        self.dropout = torch.nn.Dropout(dropout_rate)

        self.fc = torch.nn.Linear(hidden_dim, vocab_size)
        
        if tie_weights:
            assert embedding_dim == hidden_dim, 'cannot tie, check dims'
            self.embedding.weight = self.fc.weight
        #self.init_weights()

    def forward(self, src, hidden):
        embedding = self.dropout(self.embedding(src))
        output, hidden = self.lstm(embedding, hidden)          
        output = self.dropout(output) 
        prediction = self.fc(output)
        return prediction, hidden
    

    def init_weights(self):
        init_range_emb = 0.1
        init_range_other = 1/math.sqrt(self.hidden_dim)
        self.embedding.weight.data.uniform_(-init_range_emb, init_range_emb)
        self.fc.weight.data.uniform_(-init_range_other, init_range_other)
        self.fc.bias.data.zero_()
        for i in range(self.num_layers):
            self.lstm.all_weights[i][0] = torch.FloatTensor(self.embedding_dim,
                    self.hidden_dim).uniform_(-init_range_other, init_range_other) 
            self.lstm.all_weights[i][1] = torch.FloatTensor(self.hidden_dim, 
                    self.hidden_dim).uniform_(-init_range_other, init_range_other) 

    def init_hidden(self, batch_size, device):
        hidden = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        cell = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        return hidden, cell
    
    def detach_hidden(self, hidden):
        hidden, cell = hidden
        hidden = hidden.detach()
        cell = cell.detach()
        return hidden, cell