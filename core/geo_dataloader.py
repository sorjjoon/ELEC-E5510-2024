import torch
from torch.utils.data import Dataset
import scipy
import pandas as pd
import torchaudio


class GeoDataset(Dataset):
    def __init__(
        self, 
        data_path: str = "data/",
        split: str = "train",
        resample_sample_rate: int = 16000, # Standard audio sample rate 
        device: str = "cpu"
    ):
        self.transcript_df = pd.read_csv(data_path + split + ".csv")
        self.data_path = data_path
        self.resample_sample_rate = resample_sample_rate

    def __len__(self):
        return len(self.transcript_df)

    def __getitem__(self, idx):
        audio_file_path = self.data_path + self.transcript_df.iloc[idx, 0]
        transcription = self.transcript_df.iloc[idx, 1]  
        
        waveform, original_sample_rate = torchaudio.load(audio_file_path)
        waveform = torchaudio.transforms.Resample(orig_freq=original_sample_rate, new_freq=self.resample_sample_rate)(waveform)
        waveform = waveform.squeeze(0) # Remove channel dimension
        return waveform, transcription


# Handle variable waveform length in one batch using padding 
def collate_fn(itemlist):
    max_waveform_length = max(len(waveform) for waveform, _ in itemlist)
    waveform_batch = torch.zeros(len(itemlist), max_waveform_length)  
    transcript_batch = []
    
    for i, (waveform, transcript) in enumerate(itemlist):
        waveform_batch[i, :len(waveform)] = waveform 
        transcript_batch.append(transcript)  

    return waveform_batch, transcript_batch
    