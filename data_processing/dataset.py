import torch
from torch.utils.data import Dataset
from data_processing.tokenizer import encode


class PeptideDataset(Dataset):
    def __init__(self, sequences, max_len=60):
        self.data = [encode(seq, max_len) for seq in sequences]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx][:-1])
        y = torch.tensor(self.data[idx][1:])
        return x, y
