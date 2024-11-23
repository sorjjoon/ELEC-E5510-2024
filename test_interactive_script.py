# %%
import IPython
import matplotlib.pyplot as plt

import torch
import torchaudio
from torch.utils.data import DataLoader

from core.geo_dataloader import GeoDataset, collate_fn

# %%
# Load dataset and dataloader
dataset = GeoDataset(split="train")
dataloader = DataLoader(dataset, batch_size=1,
                        shuffle=True, collate_fn=collate_fn)

# %%
# Define your feature extraction transformation here
spec_transform = torchaudio.transforms.Spectrogram(
    n_fft=256, power=None, window_fn=torch.hann_window)
mel_spec_transform = torchaudio.transforms.MelSpectrogram(
    n_fft=256, n_mels=20, window_fn=torch.hann_window)
mfcc_transform = torchaudio.transforms.MFCC(n_mfcc=20)

# %%

words = set()
with open("sentences", "w", encoding="utf-8") as f:
    for features, transcript_ids, transcript in GeoDataset(split="train", transform_fn=mel_spec_transform):
        words.update(transcript.split())
        f.write(transcript+"\n")
print(len(words))
# %%
