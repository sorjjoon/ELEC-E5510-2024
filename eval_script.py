# %%
import numpy as np
import torch
from torch import nn
from torchaudio import transforms
from torch.utils.data import DataLoader
from torchmetrics.text import WordErrorRate

from core.models.CTCDecoder import GreedyCTCDecoder
from core.models.resnet_gru import SpeechRecognitionModel
from core.geo_dataloader import GeoDataset, collate_fn, get_unique_characters
from core.feature_transforms import LogMelSpec, Spectrogram, MFCC
from core.models.LMScorer import KenlmLMScorer, RNNScorer
from tqdm import tqdm

kenlm_enabled = False
try:
    import kenlm
    kenlm_enabled = True
except ImportError:
    pass






# %%
# model = SpeechRecognitionModel(n_feats=80, n_class=n_class)
model = torch.load("checkpoints/model-epoch-25.pt")
if kenlm_enabled:
    print("Using kenlm scorer from kenlm/dev-4.arpa")
    model.lm_scorer = KenlmLMScorer("kenlm/dev-4.arpa")
else:
    model.lm_scorer = None
    

# model.lm_scorer = RNNScorer(torch.load("checkpoints/lm/run-1/model-epoch-2.pt", weights_only=False))

# %%
epochs = 100

metric = WordErrorRate()
criterion = nn.CTCLoss(blank=0, zero_infinity=True)
transform = MFCC(n_mfcc=80)
# %%
test_dataset = GeoDataset(split="dev", transform_fn=transform)
test_dataloader = DataLoader(test_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

n_class = test_dataset.__num_classes__()
vocab = test_dataset.get_vocabulary()
reverse_vocab = test_dataset.get_vocabulary(reverse=True)



predicted_text = []
ground_truth_text = []
losses = []
wer = []
model.eval()
with torch.no_grad():
    for batch_id, batch in tqdm(enumerate(test_dataloader), leave=True, total=len(test_dataloader)):
        features, transcript_ids, input_lengths, target_lengths, transcripts = batch

        logits = model(features)
        loss = criterion(logits.transpose(0, 1), transcript_ids, input_lengths, target_lengths)

        predicted_text = model.score_decode_batch(logits, input_lengths, reverse_vocab)

        batch_wer = metric(predicted_text, transcripts)
        #print(predicted_text[0], transcripts[0])

        losses.append(loss.item())
        wer.append(batch_wer)            
        



# %%

print(f"WER: {np.mean(wer)}")


# %%
