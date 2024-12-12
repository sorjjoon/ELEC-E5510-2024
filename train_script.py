# %%
import pandas as pd
import os

import torch
from torch import nn
from torchaudio import transforms
from torch.utils.data import DataLoader
from torchmetrics.text import WordErrorRate, CharErrorRate

from core.models.resnet_gru import SpeechRecognitionModel
from core.geo_dataloader_ctc import GeoDataset, GeoTestDataset, collate_fn
from core.feature_transforms import LogMelSpec, Spectrogram, MFCC
from core.utils import decode_batch, greedy_decode_batch, beam_search_decode_batch, get_logger, score_decode_batch
from core.models.LMScorer import KenlmLMScorer


CHECKPOINT_DIR = "checkpoints/resnet_gru_2/run-6"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# %%
transform = MFCC(n_mfcc=40)

dataset = GeoDataset(split="train", transform_fn=transform)
dataloader = DataLoader(dataset, batch_size=32,
                        shuffle=True, collate_fn=collate_fn)

n_class = dataset.__num_classes__()
vocab = dataset.get_vocabulary()
reverse_vocab = dataset.get_vocabulary(reverse=True)

print(vocab)
print(n_class)

# %%
# model = SpeechRecognitionModel(n_feats=40, n_class=n_class)
model = torch.load("checkpoints/resnet_gru_2/run-5/model-epoch-50.pt")
# lm scorer
lm_scorer = KenlmLMScorer("kenlm/dev-4.arpa")

# %%
log = get_logger(CHECKPOINT_DIR)

epochs = 50

metric = WordErrorRate()
criterion = nn.CTCLoss(blank=0, zero_infinity=True, reduction="mean")
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-5)

for epoch in range(epochs):
    losses = []
    wer = []
    model.train()
    for batch_id, batch in enumerate(dataloader):
        features, transcript_ids, input_lengths, target_lengths, transcripts = batch

        logits = model(features)
        loss = criterion(logits.transpose(0, 1), transcript_ids,
                         input_lengths, target_lengths)

        predicted_raw_text = model.decode_batch(
            logits, input_lengths, reverse_vocab)
        predicted_text = model.greedy_decode_batch(
            logits, input_lengths, reverse_vocab)
        batch_wer = metric([predicted_text[0]], [transcripts[0]])

        losses.append(loss.item())
        wer.append(batch_wer)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (batch_id + 1) % 10 == 0:
            log._info(
                f"Batch {batch_id + 1}/{len(dataloader)}: Loss {losses[-1]} | WER: {wer[-1]}\n| Raw Predicted Word {[predicted_raw_text[0]]}\n| Decoded Predicted Word {[predicted_text[0]]}\n| Actual Word {[transcripts[0]]}\n")

    log._info(
        f"Epoch {epoch+1}/{epochs}: Average loss {sum(losses) / len(losses)} | Average WER {sum(wer) / len(wer)}")
    if (epoch + 1) % 5 == 0:
        torch.save(model, f"{CHECKPOINT_DIR}/model-epoch-{epoch+1}.pt")
        log._info("Model saved to checkpoints/\n")
    log._info("")

# %%
metric = WordErrorRate()
metric_2 = CharErrorRate()

dev_dataset = GeoDataset(split="dev", transform_fn=transform)
dev_dataloader = DataLoader(
    dev_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)

predicted_text = []
lm_predicted_text = []
# beam_predicted_text = []
ground_truth_text = []

model.eval()
with torch.no_grad():
    for batch in dev_dataloader:
        features, transcript_ids, input_lengths, target_lengths, transcripts = batch
        logits = model(features)
        # Variants of different encoding
        # beam_predicted_text += model.beam_search_decode_batch(logits, reverse_vocab)
        predicted_text += greedy_decode_batch(logits, None, reverse_vocab)
        lm_predicted_text += score_decode_batch(
            lm_scorer, logits, None, vocab, lm_weight=2.0)

        ground_truth_text += transcripts

avg_wer = metric(predicted_text, ground_truth_text)
avg_cer = metric_2(predicted_text, ground_truth_text)

print(f"WER: {avg_wer}")
print(f"CER: {avg_cer}")

# %%
pd.DataFrame({"truth": ground_truth_text, "pred": predicted_text}).to_csv(
    f"{CHECKPOINT_DIR}/test-results.csv", index=False)

# %%
test_dataset = GeoTestDataset(transform_fn=transform)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

predicted_text = []

model.eval()
with torch.no_grad():
    for features in test_dataloader:
        logits = model(features)
        predicted_text += greedy_decode_batch(logits, None, reverse_vocab)

predicted_text

# %%

df = test_dataset.df.copy()
df["transcript"] = pd.Series(predicted_text)
df.to_csv("test.csv", index=False)
