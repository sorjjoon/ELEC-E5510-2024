from torch import nn
from torchaudio import transforms
import torch


class LogMelSpec(nn.Module):
    def __init__(
        self, 
        sample_rate=16000, 
        n_mels=100, 
    ):
        super().__init__()
        self.transform = transforms.MelSpectrogram(
                            sample_rate=sample_rate, 
                            n_mels=n_mels, 
                            window_fn=torch.hann_window,
                            normalized=True,
                        )
    def forward(self, x):
        x = self.transform(x)
        # x = torch.log(x + 1e-8)
        x = x.transpose(1, 2)
        return x


class MFCC(nn.Module):
    def __init__(
        self, 
        sample_rate=16000, 
        n_mfcc=40, 
    ):
        super().__init__()
        self.transform = transforms.MFCC(
                            sample_rate=sample_rate, 
                            n_mfcc=n_mfcc, 
                            log_mels=True,
                            
                        )
    def forward(self, x):
        x = self.transform(x)
        x = (x - x.mean()) / (x.std())
        x = x.transpose(1, 2)
        return x


class Spectrogram(nn.Module):
    def __init__(
        self, 
        n_fft=512
    ):
        super().__init__()
        self.transform = transforms.Spectrogram(
                            n_fft=n_fft,
                            normalized=True,
                        )

    def forward(self, x):
        x = self.transform(x)
        x = (x - x.min()) / (x.max() - x.min())
        x = x.transpose(1, 2)
        return x