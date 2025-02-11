import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, input, gt):
        self.input = torch.tensor(input, dtype=torch.float32)
        self.gt = torch.tensor(gt, dtype=torch.float32)

    def __len__(self):
        return len(self.input)

    def __getitem__(self, idx):
        x = self.input[idx]
        y = self.gt[idx]
        return x, y
