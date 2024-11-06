# %%
import IPython
import matplotlib.pyplot as plt

import torch
import torchaudio
from torch.utils.data import DataLoader

from core.CTCDecoder import GreedyCTCDecoder
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
# Test script to observe outcome

audio_batch, transcript_batch = next(iter(dataloader))

spec_batch = spec_transform(audio_batch)
mel_spec_batch = mel_spec_transform(audio_batch)
mfcc_batch = mfcc_transform(audio_batch)

fig, axs = plt.subplots(1, 3, figsize=(18, 3))
axs[0].imshow(spec_batch[0, :, :].abs().log().numpy(),
              origin='lower', aspect="auto")
axs[0].set_title("Spectrogram features on log scale")
axs[1].imshow(mel_spec_batch[0, :, :].log().numpy(),
              origin='lower', aspect="auto")
axs[1].set_title("Mel-Spectrogram features on log scale")
axs[2].imshow(mfcc_batch[0, :, :].numpy(), origin='lower', aspect="auto")
axs[2].set_title("MFCCs features")
plt.show()

print(f"Transcription: {transcript_batch[0]}")
IPython.display.display(IPython.display.Audio(
    audio_batch[0, :].detach().numpy(), rate=int(16000)))

# %%
# Init prebuilt model
bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
model = bundle.get_model()


# %%
# Extract features (using the raw audio atm)
with torch.inference_mode():
    waveform, correct_transcript = next(iter(dataloader))
    features, _ = model.extract_features(waveform)
    emission, _ = model(waveform)

# %%
# Visualize emission
plt.imshow(emission[0].cpu().T, interpolation="nearest")
plt.title("Classification result")
plt.xlabel("Frame (time-axis)")
plt.ylabel("Class")
plt.tight_layout()
print("Class labels:", bundle.get_labels())

# %%
decoder = GreedyCTCDecoder(labels=bundle.get_labels())
transcript = decoder(emission[0])
print(
    f"Transcript {' '.join( transcript.split('|'))}, vs {' '.join(correct_transcript)}")

# %%
