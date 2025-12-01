import torch
import torch.nn as nn
from transformers import EsmModel
from config import config


class ESMMLPClassifier(nn.Module):
    def __init__(self, esm_model_name, num_labels, hidden_dim=256, dropout=0.2):
        super().__init__()
        self.esm = EsmModel.from_pretrained(esm_model_name)
        d_model = self.esm.config.hidden_size

        # 两层 MLP 分类头
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_labels),
        )

    @staticmethod
    def masked_mean(x, mask):
        """
        x: [B, L, D]
        mask: [B, L] (1 for valid, 0 for pad)
        return: [B, D]
        """
        mask = mask.unsqueeze(-1).to(x.dtype)         # [B, L, 1]
        s = (x * mask).sum(dim=1)                     # [B, D]
        d = mask.sum(dim=1).clamp(min=1.0)            # [B, 1]
        return s / d

    def forward(self, input_ids, attention_mask, labels=None, **kwargs):
        outputs = self.esm(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state  # [B, L, D]

        # 使用 mask-aware 平均池化，避免 padding 影响
        pooled = self.masked_mean(last_hidden, attention_mask)  # [B, D]
        logits = self.mlp(pooled)

        if labels is not None:
            loss = nn.functional.cross_entropy(logits, labels)
            return {"loss": loss, "logits": logits}
        return {"logits": logits}


def load_model(model_dir_or_name, num_labels):
    import os
    model = ESMMLPClassifier(
        esm_model_name=config["model_name"],
        num_labels=num_labels,
        hidden_dim=config.get("mlp_hidden_dim", 256),
        dropout=config.get("mlp_dropout", 0.2),
    )

    if model_dir_or_name and os.path.isdir(model_dir_or_name):
        state_dict_path = os.path.join(model_dir_or_name, "pytorch_model.bin")
        model.load_state_dict(torch.load(state_dict_path, map_location=config["device"]))
    return model
