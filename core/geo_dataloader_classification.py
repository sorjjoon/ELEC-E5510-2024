import torch
from torch.utils.data import Dataset
import scipy
import pandas as pd
import torchaudio


def get_unique_characters(df):
    unique_tokens = set()
    transcript = df["transcript"]
    for transcript in transcript:
        unique_tokens.update(transcript)
    sorted_tokens = sorted(list(unique_tokens)) # Ensure consistent order
    return sorted_tokens


def preprocess(df):
    cloned_df = df.copy()
    unique_tokens = get_unique_characters(cloned_df)
    non_alpha_tokens = [char for char in unique_tokens if not char.isalpha() and char != ' ']
    cloned_df['transcript'] = cloned_df['transcript'].apply(
        lambda transcript: "<" + ''.join([char for char in transcript if char not in non_alpha_tokens]) + ">"
    )
    return cloned_df


class GeoDataset(Dataset):
    def __init__(
        self, 
        transform_fn, 
        data_path: str = "data/",
        split: str = "train",
        resample_sample_rate: int = 16000, # Standard audio sample rate 
        device: str = "cpu"
    ):
        raw_transcript_df = pd.read_csv(data_path + split + ".csv", encoding="utf-8-sig")
        self.transcript_df = preprocess(raw_transcript_df)

        self.vocab = self.get_vocabulary()
        self.transform_fn = transform_fn

        self.data_path = data_path
        self.resample_sample_rate = resample_sample_rate

    def __len__(self):
        return len(self.transcript_df)
    
    def __num_classes__(self):
        unique_characters = get_unique_characters(self.transcript_df)
        return len(unique_characters)

    def __getitem__(self, idx):
        audio_file_path = self.data_path + self.transcript_df.iloc[idx, 0]
        transcript = self.transcript_df.iloc[idx, 1]  
        transcript_ids = torch.tensor(self.encode_transcript(transcript), dtype=torch.long)
        
        waveform, original_sample_rate = torchaudio.load(audio_file_path)
        waveform = torchaudio.transforms.Resample(orig_freq=original_sample_rate, new_freq=self.resample_sample_rate)(waveform)
        waveform = waveform.squeeze(0) # Remove channel dimension
        features = self.transform_fn(waveform.unsqueeze(0)).squeeze(0)
        return features, transcript_ids, transcript
    
    def encode_transcript(self, transcript):
        return [self.vocab[char] for char in transcript]
    
    def get_vocabulary(self, reverse: bool = False):
        unique_characters = get_unique_characters(self.transcript_df)
        vocab = {char: (id + 1) for id, char in enumerate(unique_characters)} if not reverse \
            else {(id + 1): char for id, char in enumerate(unique_characters)}
        return vocab
    

class GeoTestDataset(Dataset):
    def __init__(
        self, 
        transform_fn, 
        data_path: str = "data/",
        resample_sample_rate: int = 16000, # Standard audio sample rate 
        device: str = "cpu"
    ):
        self.df = pd.read_csv(data_path + "test_release.csv", encoding="utf-8-sig")
        self.transform_fn = transform_fn
        self.data_path = data_path
        self.resample_sample_rate = resample_sample_rate

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        audio_file_path = self.data_path + self.df.iloc[idx, 0]
        waveform, original_sample_rate = torchaudio.load(audio_file_path)
        waveform = torchaudio.transforms.Resample(orig_freq=original_sample_rate, new_freq=self.resample_sample_rate)(waveform)
        waveform = waveform.squeeze(0) # Remove channel dimension
        features = self.transform_fn(waveform.unsqueeze(0)).squeeze(0)
        return features


# Handle variable waveform length in one batch using padding 
def collate_fn(itemlist):
    max_feature_length = max(feature.shape[0] for feature, _, _ in itemlist)
    feature_dim = itemlist[0][0].shape[1]
    feature_batch = torch.zeros(len(itemlist), max_feature_length, feature_dim)  

    max_transcript_length = max(transcript_id.shape[0] for _, transcript_id, _ in itemlist)
    transcript_id_batch = torch.zeros(len(itemlist), max_transcript_length, dtype=torch.long) 
    transcript_batch = []

    input_lengths = torch.zeros(len(itemlist), dtype=torch.long)  
    target_lengths = torch.zeros(len(itemlist), dtype=torch.long)  

    for i, (feature, transcript_id, transcript) in enumerate(itemlist):
        feature_batch[i, :feature.shape[0], :] = feature 
        transcript_id_batch[i, :transcript_id.shape[0]] = transcript_id 
        
        input_lengths[i] = feature.shape[0] 
        target_lengths[i] = transcript_id.shape[0]
        transcript_batch.append(transcript)

    return feature_batch, transcript_id_batch, input_lengths, target_lengths, transcript_batch
    