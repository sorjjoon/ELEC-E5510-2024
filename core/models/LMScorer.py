import kenlm
kenlm_enabled = False
try:
    import kenlm
    kenlm_enabled = True
except ImportError:
    pass

import typing
# FIXME, python is dumb
from .RNN import T_EOF, T_START, T_UNKOWN, LmRNN

T_UNKOWN = "<unk>"
T_EOF = "<eof>"
T_START = "<start>"

import torch
import numpy as np

class KenlmLMScorer:
    def __init__(self, path):
        if kenlm_enabled:
             self.model = kenlm.LanguageModel(path)
        else:
            self.model = None

    def score(self, sentence):
        return self.model.score(sentence, bos = True, eos = True)
    

class RNNScorer:
    def __init__(self, model: LmRNN):
        vocab = model.vocab

        self.model = model
        self.vocab:typing.Dict[typing.Union[int, str], typing.Union[int, str]] = {}
        self.vocab.update(vocab)

        # Add reverse mappings, to make lookups easier
        for word, index in vocab.items():
            self.vocab[index] = word

    def score(self, sentence):
        # Predict the probability of the full sentence
        # Init state, assume sentence does not contain our eof and start tokens
        assert T_START not in sentence
        assert T_EOF not in sentence



        tokens = [T_START, *sentence.split(), T_EOF]
        probs = self.get_probability(tokens, "cpu")
        
        # Probs are not logarithmic at this point
        return np.log(np.prod(probs))


    def get_probability(self, tokens,  device)-> typing.List[float]:
        """
            Note, tokens need to include START and EOF
        """
        self.model.eval()
        

        # Note, we can use a hard coded batch_size of 1
        batch_size = 1

        UNKNWON_VAL = self.vocab[T_UNKOWN]

        hidden = self.model.init_hidden(batch_size, device)

        with torch.no_grad():
            probabilites = []
            indices = []
            for t in tokens:
                indice = self.vocab.get(t, UNKNWON_VAL)
                indices.append(indice)

                src = torch.LongTensor([indices]).to(device)
                
                prediction, hidden = self.model(src, hidden)

                # Could also add a layer for this
                probs = torch.softmax(prediction[:, -1], dim=-1)  
                probabilites.append(probs[0, indice].item())
        
        return probabilites
                
