import torch
from torch.utils.data import Dataset


class AGNewsDataset(Dataset):
    def __init__(self, input_ids, attention_masks, labels):
        self.i = input_ids
        self.a = attention_masks
        self.labels = labels

    def __len__(self):
        return len(self.i)

    def __getitem__(self, idx):
        return {
            'input_ids': torch.tensor(self.i[idx], dtype=torch.long),
            'attention_mask': torch.tensor(self.a[idx], dtype=torch.long),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }
