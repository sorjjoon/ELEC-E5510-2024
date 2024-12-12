import torch
from torchaudio.models.decoder import ctc_decoder
import logging
import os

from torch.distributions import Categorical


class DualLogger:
    def __init__(self, dir_path: str):
        # Create the logs directory if it doesn't exist
        os.makedirs(f"{dir_path}/logs", exist_ok=True)

        # Initialize a logger instance specific to this class
        self.logger = logging.getLogger(
            "DualLogger")  # Use a unique logger name
        self.logger.setLevel(logging.INFO)

        # Check if handlers already exist to avoid duplicate messages
        if not self.logger.hasHandlers():
            # File handler for logging to a file
            file_handler = logging.FileHandler(
                f"{dir_path}/logs/training.log", mode='w', encoding='utf-8-sig')
            file_handler.setFormatter(
                logging.Formatter('%(asctime)s:\n%(message)s'))

            # Console handler for logging to console
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(logging.Formatter('%(message)s'))

            # Add handlers to the logger
            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)

    def _info(self, msg):
        self.logger.info(msg)


def get_logger(dir_path):
    return DualLogger(dir_path)


def decode_batch(logits, vocab, input_length=None, ignore_index=0):
    pred_ids = torch.argmax(logits, dim=-1)
    decoded_texts = []
    for i in range(logits.shape[0]):
        unique_ids = pred_ids[i]
        if input_length != None:
            unique_ids = unique_ids[-input_length[i]:]
        unique_ids = [id for id in unique_ids if id != ignore_index]
        decoded_text = "".join([vocab[id.item()] for id in unique_ids])
        decoded_texts.append(decoded_text.strip())
    return decoded_texts


def greedy_decode_batch(logits, input_length, vocab, ignore_index=0):
    pred_ids = torch.argmax(logits, dim=-1)
    decoded_texts = []
    for i in range(logits.shape[0]):
        if input_length != None:
            unique_ids = torch.unique_consecutive(
                pred_ids[i][-input_length[i]:])
        else:
            unique_ids = torch.unique_consecutive(pred_ids[i])
        unique_ids = [id for id in unique_ids if id != ignore_index]
        decoded_text = "".join([vocab[id.item()] for id in unique_ids])
        decoded_texts.append(decoded_text.strip())
    return decoded_texts


def beam_search_decode_batch(logits, vocab):
    decoded_texts = []
    decoder = ctc_decoder(
        lexicon=None,
        tokens="tokens.txt",  # Vocabulary
        lm="checkpoints/lm/dev-4.binary",
        lm_weight=100,
        blank_token="_",
        sil_token='|',
        beam_size=50,
    )
    for i in range(logits.shape[0]):
        results = decoder(logits[i, :].unsqueeze(0))
        tokens = results[0][0].tokens
        decoded_text = "".join([vocab[id.item()] for id in tokens])
        decoded_texts.append(decoded_text.strip())
    return decoded_texts


def score_decode_batch(lm_scorer, logits, input_length, vocab, k=100, lm_weight=0.7, include_best_acoustic=False):
    if lm_scorer is None or lm_weight is None or lm_weight == 0:
        return greedy_decode_batch(logits, input_length, vocab)

    def decode_text(pred_ids):
        if input_length != None:
            unique_ids = torch.unique_consecutive(
                pred_ids[i][-input_length[i]:])
        else:
            unique_ids = torch.unique_consecutive(pred_ids[i])

        unique_ids = [id for id in unique_ids if id != 0]

        return "".join([vocab[id.item()] for id in unique_ids])

    def do_score(topk_ids, i, text):
        prob = logits[i].gather(1, topk_ids.unsqueeze(-1)).squeeze(-1)

        if input_length != None:
            acoustic_score = prob[-input_length[i]:].sum().item()
        else:
            acoustic_score = prob.sum().item()

        lm_score = lm_scorer.score(text)
        return (acoustic_score + lm_weight * lm_score, text)

    decoded_texts = []

    if include_best_acoustic:
        best_predict = torch.argmax(logits, dim=-1)

    dist = Categorical(logits=logits)
    samples = dist.sample((k,))

    for i in range(logits.shape[0]):
        seen = set()
        sequences = []

        # Make sure to sample the top acoustic prediction
        if include_best_acoustic:
            t = decode_text(best_predict[i])
            seen.add(t)
            sequences.append(do_score(best_predict[i], i, t))

        for j in range(k):
            # Sample from top predictions
            text = decode_text(samples[j, i])
            if text in seen:
                continue

            seen.add(text)
            seq = do_score(samples[j, i], i, text)
            sequences.append(seq)

        # Sort based on score
        sequences.sort(reverse=True)
        sentence = sequences[0][1]
        decoded_texts.append(sentence.strip())
    return decoded_texts
