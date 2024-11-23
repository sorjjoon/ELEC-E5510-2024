# %%
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
kenlm_enabled = False
try:
    import kenlm
    kenlm_enabled = True
except ImportError:
    pass



# %%
transform = MFCC(n_mfcc=80)
# transform = Spectrogram(n_fft=512)

dataset = GeoDataset(split="train", transform_fn=transform)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)

n_class = dataset.__num_classes__()
vocab = dataset.get_vocabulary()
reverse_vocab = dataset.get_vocabulary(reverse=True)

print(vocab)
# %%
# model = SpeechRecognitionModel(n_feats=80, n_class=n_class)
model = torch.load("checkpoints/run-2/model-epoch-50.pt")
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
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(epochs):
    losses = []
    wer = []
    model.train()
    for batch_id, batch in enumerate(dataloader):
        features, transcript_ids, input_lengths, target_lengths, transcripts = batch

        logits = model(features)
        loss = criterion(logits.transpose(0, 1), transcript_ids, input_lengths, target_lengths)

        predicted_raw_text = model.decode_batch(logits, input_lengths, reverse_vocab)
        predicted_text = model.greedy_decode_batch(logits, input_lengths, reverse_vocab)
        batch_wer = metric([predicted_text[0]], [transcripts[0]])

        losses.append(loss.item())
        wer.append(batch_wer)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (batch_id + 1) % 10 == 0:
            print(f"Batch {batch_id + 1}/{len(dataloader)}: Loss {losses[-1]} | WER: {wer[-1]}\n| Raw Predicted Word {[predicted_raw_text[0]]}\n| Decoded Predicted Word {[predicted_text[0]]}\n| Actual Word {[transcripts[0]]}\n")

    print(f"Epoch {epoch+1}/{epochs}: Average loss {sum(losses) / len(losses)} | Average WER {sum(wer) / len(wer)}")
    torch.save(model, f"checkpoints/run-3/model-epoch-{epoch+1}.pt")
    print("Model saved to checkpoints/\n")

# %%
test_dataset = GeoDataset(split="dev", transform_fn=transform)
test_dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

predicted_text = []
ground_truth_text = []

model.eval()
with torch.no_grad():
    for batch in test_dataloader:
        features, transcript_ids, input_lengths, target_lengths, transcripts = batch
        logits = model(features)
        predicted_text += model.greedy_decode_batch(logits, input_lengths, reverse_vocab)
        ground_truth_text += transcripts
        print(predicted_text)
        print(ground_truth_text)
        break

avg_wer = metric(predicted_text, ground_truth_text)
# print(f"WER: {avg_wer}")

# %%
transformed_values = dataset[0][0].unsqueeze(0)
# labels = torch.tensor([vocab[char] for char in dataset[0][1]]).unsqueeze(0)

print(transformed_values.shape)
logits = model(transformed_values)
print(logits.shape)
text = model.decode_batch(logits, [300], reverse_vocab)
print(text)
# %%
vocab