import kenlm
kenlm_enabled = False
try:
    import kenlm
    kenlm_enabled = True
except ImportError:
    pass

class KenlmLMScorer:
    def __init__(self, path):
        if kenlm_enabled:
             self.model = kenlm.LanguageModel(path)
        else:
            self.model = None

    def score(self, sentence):
        return self.model.score(sentence, bos = True, eos = True)