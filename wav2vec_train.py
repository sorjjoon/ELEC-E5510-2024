# -*- coding: utf-8 -*-


from transformers import Wav2Vec2ForCTC
from transformers import Trainer
from transformers import TrainingArguments
import random
import IPython.display as ipd
import os
from datasets import load_dataset
from datasets import load_dataset, Audio
from torch.utils.data import DataLoader
import torchaudio
from evaluate import load
import numpy as np
from torch import nn
from torchaudio import transforms
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Wav2Vec2FeatureExtractor, Wav2Vec2CTCTokenizer, BatchFeature
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
import torch
from tqdm import tqdm, trange
import json
import re
from transformers import Wav2Vec2FeatureExtractor
import sys
# %%
# from google.colab import drive
# drive.mount('/content/gdrive/')

# sys.path.append('/content/gdrive/content/drive/MyDrive/Colab Notebooks/speech-reco')

# os.chdir("/content/gdrive/MyDrive/Colab Notebooks/speech-reco/")

#!pip install evaluate
#!pip install jiwer


# %%


#  Copied from
#  https://github.com/huggingface/transformers/blob/9a06b6b11bdfc42eea08fa91d0c737d1863c99e3/examples/research_projects/wav2vec2/run_asr.py#L81

@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        max_length_labels (:obj:`int`, `optional`):
            Maximum length of the ``labels`` returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        input_features = [{"input_values": feature["input_values"]}
                          for feature in features]
        label_features = [{"input_ids": feature["labels"]}
                          for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                max_length=self.max_length_labels,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch


# from torchmetrics.text import WordErrorRate


# Preprocessing the datasets.

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

!nvidia-smi

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


rand_int = random.randint(0, len(train_set)-1)

ipd.Audio(data=train_set[rand_int]["audio"]
          ["array"], autoplay=True, rate=16000)

# %%


def prepare_dataset(batch):
    audio = batch["audio"]

    # batched output is "un-batched"
    batch["input_values"] = processor(
        audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]

    with processor.as_target_processor():
        batch["labels"] = processor(batch["transcript"]).input_ids
    return batch


train_set = train_set.map(
    prepare_dataset, remove_columns=train_set.column_names)
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


training_args = TrainingArguments(
    output_dir="./wav2vec2-run1",
    group_by_length=True,
    report_to="none",
    per_device_train_batch_size=16,
    gradient_accumulation_steps=2,
    evaluation_strategy="steps",
    num_train_epochs=30,
    fp16=True,
    save_steps=100,
    eval_steps=100,
    logging_steps=10,
    learning_rate=3e-4,
    warmup_steps=500,
    save_total_limit=2,
)


trainer = Trainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_set,
    eval_dataset=eval_set,
    tokenizer=processor.feature_extractor,
)
os.environ["WANDB_DISABLED"] = "true"
trainer.train(resume_from_checkpoint=True)
