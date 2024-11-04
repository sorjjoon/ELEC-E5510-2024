# %%
import IPython
import matplotlib.pyplot as plt

import torch
from torchaudio import transforms
from torch.utils.data import DataLoader

from core.geo_dataloader import GeoDataset, collate_fn

# %%
# Load dataset and dataloader
dataset = GeoDataset(split="train")
dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)

# %%
# Define your feature extraction transformation here
spec_transform = transforms.Spectrogram(n_fft=256, power=None, window_fn=torch.hann_window)
mel_spec_transform = transforms.MelSpectrogram(n_fft=256, n_mels=20, window_fn=torch.hann_window) 
mfcc_transform = transforms.MFCC(n_mfcc=20) 

# %%
# Test script to observe outcome

iterable_loader = iter(dataloader)
audio_batch, transcript_batch = next(iterable_loader)

spec_batch = spec_transform(audio_batch)
mel_spec_batch = mel_spec_transform(audio_batch)
mfcc_batch = mfcc_transform(audio_batch)

fig, axs = plt.subplots(1, 3, figsize=(18, 3))
axs[0].imshow(spec_batch[0, :, :].abs().log().numpy(), origin='lower', aspect="auto")
axs[0].set_title("Spectrogram features on log scale")
axs[1].imshow(mel_spec_batch[0, :, :].log().numpy(), origin='lower', aspect="auto")
axs[1].set_title("Mel-Spectrogram features on log scale")
axs[2].imshow(mfcc_batch[0, :, :].numpy(), origin='lower', aspect="auto")
axs[2].set_title("MFCCs features")
plt.show()

print(f"Transcription: {transcript_batch[0]}")
IPython.display.display(IPython.display.Audio(audio_batch[0, :].detach().numpy(), rate=int(16000)))

# %%