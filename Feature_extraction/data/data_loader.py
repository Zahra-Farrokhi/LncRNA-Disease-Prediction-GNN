# data/data_loader.py

from Bio import SeqIO
import numpy as np
import logging
import torch
from torch.utils.data import Dataset
def load_sequences(fasta_file, min_length, valid_nucleotides={'A', 'C', 'T', 'G'}):
    sequences = []
    try:
        for record in SeqIO.parse(fasta_file, "fasta"):
            seq = str(record.seq).upper()
            if len(seq) > min_length and set(seq).issubset(valid_nucleotides):
                sequences.append(seq)
    except Exception as e:
        logging.error(f"Error while parsing FASTA file: {e}")
        raise e
    logging.info(f"Total valid sequences: {len(sequences)}")
    return sequences

def one_hot_encode(sequence, mapping={'A': 0, 'C': 1, 'T': 2, 'G': 3}, input_size=4):
    one_hot = np.zeros((len(sequence), input_size), dtype=np.float32)
    for i, nucleotide in enumerate(sequence):
        if nucleotide in mapping:
            one_hot[i, mapping[nucleotide]] = 1.0
    return one_hot

def preprocess_data(fasta_file, sequence_length):
    logging.info("Loading sequences...")
    sequences = load_sequences(fasta_file, min_length=sequence_length)
    logging.info("One-hot encoding sequences...")
    encoded_sequences = [one_hot_encode(seq) for seq in sequences]
    logging.info(f"Total encoded sequences with sufficient length: {len(encoded_sequences)}")
    return encoded_sequences

class LNCRNADataset(Dataset):
    def __init__(self, encoded_sequences, sequence_length, step=10):
        self.encoded_sequences = encoded_sequences
        self.sequence_length = sequence_length
        self.step = step
        self.sequence_sample_counts = []
        for seq in self.encoded_sequences:
            num_samples = (len(seq) - self.sequence_length) // self.step
            self.sequence_sample_counts.append(num_samples)
        self.cumulative_samples = np.cumsum([0] + self.sequence_sample_counts)
        self.total_samples = self.cumulative_samples[-1]
        logging.info(f"Initialized LNCRNADataset with {self.total_samples} total samples.")

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        seq_idx = np.searchsorted(self.cumulative_samples, idx, side='right') - 1
        sample_idx = idx - self.cumulative_samples[seq_idx]
        start_pos = sample_idx * self.step
        end_pos = start_pos + self.sequence_length
        x = self.encoded_sequences[seq_idx][start_pos:end_pos]
        y = np.argmax(self.encoded_sequences[seq_idx][end_pos])
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)
