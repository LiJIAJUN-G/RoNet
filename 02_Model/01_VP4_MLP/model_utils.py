import torch
import torch.nn as nn
from transformers import EsmModel
from config import config


class ESMMLPClassifier(nn.Module):
    def __init__(self, esm_model_name, num_labels, hidden_dim=256, dropout=0.3):
        super().__init__()
        self.esm = EsmModel.from_pretrained(esm_model_name)

        # MLP 层：两层全连接网络
        self.mlp = nn.Sequential(
            nn.Linear(self.esm.config.hidden_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_labels)
        )

    def forward(self, input_ids, attention_mask, labels=None, **kwargs):
        outputs = self.esm(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state  # [B, L, D]

        # 简单池化（取平均）
        pooled = torch.mean(last_hidden, dim=1)  # [B, D]

        logits = self.mlp(pooled)

        if labels is not None:
            loss = torch.nn.functional.cross_entropy(logits, labels)
            return {"loss": loss, "logits": logits}
        return {"logits": logits}


def load_model(model_dir_or_name, num_labels):
    import os
    # 初始化模型结构（包括 ESM 的加载）
    model = ESMMLPClassifier(
        esm_model_name=config["model_name"],
        num_labels=num_labels,
        hidden_dim=config.get("mlp_hidden_dim", 256),
        dropout=config.get("mlp_dropout", 0.3)
    )

    # 如果是训练好的模型路径，加载权重
    if model_dir_or_name and os.path.isdir(model_dir_or_name):
        state_dict_path = os.path.join(model_dir_or_name, "pytorch_model.bin")
        model.load_state_dict(torch.load(state_dict_path, map_location=config["device"]))
    return model
