# %%
import os
os.environ['TRANSFORMERS_CACHE']  = os.environ['HF_HOME'] = 'F:\\TEMP'

# %%
from transformers import Wav2Vec2FeatureExtractor
import re
import json
from tqdm import tqdm, trange
from core.Wav2VecDataCollator import DataCollatorCTCWithPadding
from core.models.LMScorer import KenlmLMScorer, RNNScorer
from core.feature_transforms import LogMelSpec, Spectrogram, MFCC
from core.models.resnet_gru import SpeechRecognitionModel
from core.models.CTCDecoder import GreedyCTCDecoder
from torchmetrics.text import WordErrorRate
from torchaudio import transforms
from torch import nn
import numpy as np
import torch

from evaluate import load

import torchaudio
from torch.utils.data import DataLoader
from datasets import load_dataset, Audio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Wav2Vec2FeatureExtractor, Wav2Vec2CTCTokenizer, BatchFeature 

from core.geo_dataloader import GeoDataset


from core.geo_dataloader import GeoDataset, collate_fn, get_unique_characters
# Preprocessing the datasets.
# We need to read the aduio files as arrays


from datasets import load_dataset


# %%
dataset = load_dataset(
    "data",
    data_files={"dev": "dev.csv", "train": "train.csv"})

train_set = dataset.get("train")
eval_set = dataset.get("dev")


# %%
chars_to_ignore_regex = '[\(\)\,\?\.\!\-\;\:\"\“\%\‘\”\—\–\„\’\«\»]'


def remove_special_characters(batch):
    batch["transcript"] = re.sub(
        chars_to_ignore_regex, '', batch["transcript"]).lower() + " "
    return batch


train_set = train_set.map(remove_special_characters)
eval_set = eval_set.map(remove_special_characters)

# %%

vocab = {}
# Delimiter
vocab["|"] = len(vocab)
vocab["<unk>"] = len(vocab)
vocab["<pad>"] = len(vocab)


for batch in train_set:
    for char in batch["transcript"]:
        if char not in vocab and not char.isspace():
            vocab[char] = len(vocab)

# %%
with open('checkpoints/vocab.json', 'w') as vocab_file:
    json.dump(vocab, vocab_file)

tokenizer = Wav2Vec2CTCTokenizer(
    "checkpoints/vocab.json", unk_token="<unk>", pad_token="<pad>", word_delimiter_token="|")

# %%

feature_extractor = Wav2Vec2FeatureExtractor(
    feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=True)


processor = Wav2Vec2Processor(
    feature_extractor=feature_extractor, tokenizer=tokenizer)

# %%
# Load audio
def add_audio(batch):
    batch["audio"] = "data/"+batch["file"]
    return batch

train_set = train_set.map(add_audio).cast_column("audio", Audio())
eval_set = eval_set.map(add_audio).cast_column("audio", Audio())

# %%
# import IPython.display as ipd
# import numpy as np
# import random

# rand_int = random.randint(0, len(train_set)-1)

# ipd.Audio(data=train_set[rand_int]["audio"]["array"], autoplay=True, rate=16000)
# %%
def prepare_dataset(batch):
    audio = batch["audio"]

    # batched output is "un-batched"
    batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
    
    with processor.as_target_processor():
        batch["labels"] = processor(batch["transcript"]).input_ids
    return batch
train_set = train_set.map(prepare_dataset, remove_columns=train_set.column_names)
eval_set = eval_set.map(prepare_dataset, remove_columns=eval_set.column_names)

# %%
data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

# %%
wer_metric = load("wer")


def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    # we do not want to group tokens when computing the metrics
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    wer = wer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}

# %%

from transformers import Wav2Vec2ForCTC

model = Wav2Vec2ForCTC.from_pretrained(
    "facebook/wav2vec2-large-xlsr-53", 
    attention_dropout=0.1,
    hidden_dropout=0.1,
    feat_proj_dropout=0.0,
    mask_time_prob=0.05,
    layerdrop=0.1,
    ctc_loss_reduction="mean", 
    pad_token_id=processor.tokenizer.pad_token_id,
    vocab_size=len(processor.tokenizer)
)
# Source https://arxiv.org/pdf/2006.13979
model.freeze_feature_extractor()
model.gradient_checkpointing_enable()


# %%
from transformers import TrainingArguments

training_args = TrainingArguments(
  output_dir="./wav2vec2-run1",
  group_by_length=True,
  per_device_train_batch_size=16,
  gradient_accumulation_steps=2,
  evaluation_strategy="steps",
  num_train_epochs=30,
  #fp16=True,
  save_steps=100,
  eval_steps=100,
  logging_steps=10,
  learning_rate=3e-4,
  warmup_steps=500,
  save_total_limit=2,
  resume_from_checkpoint=True
)


from transformers import Trainer

trainer = Trainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_set,
    eval_dataset=eval_set,
    tokenizer=processor.feature_extractor,
)
# %%
trainer.train()


# %%
