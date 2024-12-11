import torch
from torchaudio.models.decoder import ctc_decoder
import logging
import os

class DualLogger:
    def __init__(self, dir_path: str):
        # Create the logs directory if it doesn't exist
        os.makedirs(f"{dir_path}/logs", exist_ok=True)

        # Initialize a logger instance specific to this class
        self.logger = logging.getLogger("DualLogger")  # Use a unique logger name
        self.logger.setLevel(logging.INFO)

        # Check if handlers already exist to avoid duplicate messages
        if not self.logger.hasHandlers():
            # File handler for logging to a file
            file_handler = logging.FileHandler(f"{dir_path}/logs/training.log", mode='w', encoding='utf-8-sig')
            file_handler.setFormatter(logging.Formatter('%(asctime)s:\n%(message)s'))

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


def decode_batch(logits, vocab, input_length = None, ignore_index = 0):
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


def greedy_decode_batch(logits, input_length, vocab, ignore_index = 0):
    pred_ids = torch.argmax(logits, dim=-1)
    decoded_texts = []  
    for i in range(logits.shape[0]):
        if input_length != None:
            unique_ids = torch.unique_consecutive(pred_ids[i][-input_length[i]:])
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