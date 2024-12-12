from contextlib import redirect_stdout
import math
import os
import sys
import numpy as np
import torch
import typing
import kenlm
kenlm_enabled = False
try:
    import kenlm
    kenlm_enabled = True
except ImportError:
    pass

# FIXME, python is dumb

T_UNKOWN = "<unk>"
T_EOF = "<eof>"
T_START = "<start>"


class KenlmLMScorer:
    def __init__(self, path):
        # prevent debug spam
        with open(os.devnull, "a") as f:
            with redirect_stdout(f):
                if kenlm_enabled:
                    self.model = kenlm.LanguageModel(path)
                else:
                    self.model = None

    def score(self, sentence):
        log_score = self.model.score(sentence, bos=True, eos=True)
        # Convert to natural log
        return log_score * math.log(10)
