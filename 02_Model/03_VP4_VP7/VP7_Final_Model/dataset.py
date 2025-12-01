from torch.utils.data import Dataset
import torch

class HostDataset(Dataset):
    def __init__(self, csv_path, tokenizer, max_len, label2idx=None):
        import pandas as pd
        self.data = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.max_len = max_len

        if label2idx is None:
            label2idx = {label: i for i, label in enumerate(sorted(self.data["Host"].unique()))}
        self.label2idx = label2idx
        self.idx2label = {i: label for label, i in self.label2idx.items()}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq = self.data.iloc[idx]["Seq"]
        label = self.label2idx[self.data.iloc[idx]["Host"]]
        tokens = self.tokenizer(seq, return_tensors="pt", padding='max_length',
                                truncation=True, max_length=self.max_len)
        tokens = {k: v.squeeze(0) for k, v in tokens.items()}
        tokens["labels"] = torch.tensor(label)
        return tokens
