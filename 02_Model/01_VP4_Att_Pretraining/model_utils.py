import torch
import torch.nn as nn
from transformers import EsmModel
from config import config
import os

class ESMAttentionClassifier(nn.Module):
    def __init__(self, esm_model_name, num_labels, dropout=0.3):
        super().__init__()
        self.esm = EsmModel.from_pretrained(esm_model_name)
        hidden_size = self.esm.config.hidden_size

        self.attention = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, labels=None, **kwargs):
        outputs = self.esm(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state  # [B, L, D]

        # Attention pooling
        attn_weights = torch.softmax(self.attention(last_hidden).squeeze(-1), dim=1)  # [B, L]
        pooled = torch.sum(last_hidden * attn_weights.unsqueeze(-1), dim=1)  # [B, D]
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)

        if labels is not None:
            loss = torch.nn.functional.cross_entropy(logits, labels)
            return {"loss": loss, "logits": logits}
        return {"logits": logits}


def load_model(model_dir_or_name, num_labels):
    # 始终用训练前指定的 esm_model 初始化结构
    model = ESMAttentionClassifier(
        esm_model_name=config["model_name"],
        num_labels=num_labels,
        dropout=config.get("lstm_dropout", 0.1)
    )

    # 如果传入了模型权重路径，则加载 state_dict
    if model_dir_or_name and os.path.isdir(model_dir_or_name):
        weight_path = os.path.join(model_dir_or_name, "pytorch_model.bin")
        model.load_state_dict(torch.load(weight_path, map_location=config["device"]))

    return model
