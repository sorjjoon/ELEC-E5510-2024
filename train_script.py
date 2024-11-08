# %%
import torch
from torch import nn
from torchaudio import transforms
from torch.utils.data import DataLoader
from torchmetrics.text import WordErrorRate

from core.models.CTCDecoder import GreedyCTCDecoder
from core.models.resnet_gru import Encoder, SimpleEncoder
from core.geo_dataloader import GeoDataset, collate_fn, get_unique_characters
from core.feature_transforms import LogMelSpec, Spectrogram

# %%
transform = LogMelSpec(n_fft=512, n_mels=128)
# transform = Spectrogram(n_fft=512)

dataset = GeoDataset(split="train", transform_fn=transform)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)

n_classes = dataset.__num_classes__()
vocab = dataset.get_vocabulary()
reverse_vocab = dataset.get_vocabulary(reverse=True)

# %%
model = SimpleEncoder(n_features=128, n_classes=n_classes, hidden_size=256, num_rnn_layers=3)

# %%
epochs = 1

metric = WordErrorRate()
criterion = nn.CTCLoss(blank=0, zero_infinity=True, reduction="sum")
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(epochs):
    losses = []
    wer = []
    model.train()
    for batch_id, batch in enumerate(dataloader):
        features, transcript_ids, input_lengths, target_lengths, transcripts = batch
        batch_size = features.shape[0]

        logits = model(features)
        pred_lengths = torch.LongTensor([logits.shape[1]] * batch_size)
        loss = criterion(logits.transpose(0, 1), transcript_ids, input_lengths, target_lengths)
        loss /= batch_size

        predicted_text = model.decode_batch(logits[0,:].unsqueeze(0), reverse_vocab)
        batch_wer = metric(predicted_text, [transcripts[0]])

        losses.append(loss.item())
        wer.append(batch_wer)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (batch_id + 1) % 10 == 0:
            print(f"Batch {batch_id + 1}/{len(dataloader)}: Loss {losses[-1]} | WER: {wer[-1]} | Predicted Word {predicted_text} | Actual Word {[transcripts[0]]}")

    print(f"Epoch {epoch}/{epochs}: Average loss {sum(losses) / len(losses)} | Average WER {sum(wer) / len(wer)}")
    torch.save(model, "test-model.pt")
    print("Model saved to test-model.pt.")

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
        predicted_text += model.decode_batch(logits, reverse_vocab)
        ground_truth_text += transcripts
        print(predicted_text)
        print(ground_truth_text)
        break

avg_wer = metric(predicted_text, ground_truth_text)
# print(f"WER: {avg_wer}")

# %%
transformed_values = dataset[1][0].unsqueeze(0)
print(transformed_values)
# labels = torch.tensor([vocab[char] for char in dataset[0][1]]).unsqueeze(0)

print(transformed_values.shape)
logits = model(transformed_values)
print(logits.shape)

# %%
print(logits.shape)
print(torch.tensor(transformed_values.shape[1]).unsqueeze(0))
criterion(logits.transpose(0, 1), labels, torch.tensor(logits.shape[1]).unsqueeze(0), torch.tensor(labels.shape[1]).unsqueeze(0))

# %%
vocab