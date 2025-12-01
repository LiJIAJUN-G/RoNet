import torch
import torch.nn as nn
from transformers import EsmModel
from config import config


class SEBlock(nn.Module):
    def __init__(self, dim, reduction=16):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(dim, dim // reduction, 1),
            nn.ReLU(),
            nn.Conv1d(dim // reduction, dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):  # x: [B, D, L]
        scale = self.se(x)  # [B, D, 1]
        return x * scale    # 广播相乘 [B, D, L]


class ESMSEClassifier(nn.Module):
    def __init__(self, esm_model_name, num_labels, dropout=0.2):
        super().__init__()
        self.esm = EsmModel.from_pretrained(esm_model_name)
        hidden_size = self.esm.config.hidden_size

        self.se_block = SEBlock(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size * 2, num_labels)  # 拼接后为2D维度

    def forward(self, input_ids, attention_mask, labels=None, **kwargs):
        outputs = self.esm(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state  # [B, L, D]

        # Apply SE block
        x = last_hidden.permute(0, 2, 1)     # [B, D, L]
        x = self.se_block(x)                 # [B, D, L]
        x = x.permute(0, 2, 1)               # [B, L, D]

        mean_pool = x.mean(dim=1)            # [B, D]
        max_pool = x.max(dim=1).values       # [B, D]
        fused = torch.cat([mean_pool, max_pool], dim=1)  # [B, 2D]

        logits = self.classifier(self.dropout(fused))  # [B, num_labels]

        if labels is not None:
            loss = torch.nn.functional.cross_entropy(logits, labels)
            return {"loss": loss, "logits": logits}
        return {"logits": logits}


def load_model(model_dir_or_name, num_labels):
    import os
    model = ESMSEClassifier(
        esm_model_name=config["model_name"],
        num_labels=num_labels,
        dropout=0.2,
    )

    if model_dir_or_name and os.path.isdir(model_dir_or_name):
        state_dict_path = os.path.join(model_dir_or_name, "pytorch_model.bin")
        model.load_state_dict(torch.load(state_dict_path, map_location=config["device"]))
    return model
