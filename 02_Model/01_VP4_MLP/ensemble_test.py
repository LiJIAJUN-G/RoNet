import os
import torch
import numpy as np
import pandas as pd
from transformers import EsmTokenizer, Trainer
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from model_utils import load_model
from config import config


def load_label2idx_from_train(train_path):
    train_df = pd.read_csv(train_path)
    labels = sorted(train_df["Host"].unique())
    return {label: i for i, label in enumerate(labels)}

def predict_probs(seq_list, tokenizer, max_len, model_dir, num_labels, batch_size=32):
    model = load_model(model_dir, num_labels=num_labels).to(config["device"])
    model.eval()

    all_probs = []

    for i in range(0, len(seq_list), batch_size):
        batch_seqs = seq_list[i:i+batch_size]
        tokenized = tokenizer(batch_seqs, padding="max_length", truncation=True,
                              max_length=max_len, return_tensors="pt")
        tokenized = {k: v.to(config["device"]) for k, v in tokenized.items()}

        with torch.no_grad():
            outputs = model(**tokenized)
            if isinstance(outputs, dict):
                logits = outputs["logits"]
            else:
                logits = outputs.logits  # fallback for huggingface 原模型兼容
            probs = torch.nn.functional.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)

    return np.vstack(all_probs)


def ensemble_test():
    df = pd.read_csv("data/VP4_VP7/test.csv")
    vp4_tokenizer = EsmTokenizer.from_pretrained(config["model_name"])
    vp7_tokenizer = EsmTokenizer.from_pretrained(config["model_name"])

    # ✅ 使用总的 train.csv 提取标签映射
    global_train_path = "data/VP4/train.csv"
    label2idx = load_label2idx_from_train(global_train_path)
    idx2label = {i: label for label, i in label2idx.items()}
    df["label_id"] = df["Host"].map(label2idx)

    all_preds = []
    all_labels = df["label_id"].values
    label_list = list(label2idx.keys())

    for fold in range(1, 6):
        print(f"== Fold {fold} ==")

        # VP4 预测
        vp4_model_dir = os.path.join(config["save_dir"], "VP4", f"fold{fold}", config["best_model_path"])
        vp4_probs = predict_probs(
            df["VP4_Seq"].tolist(), vp4_tokenizer, config["max_length"]["VP4"],
            vp4_model_dir, num_labels=len(label2idx)
        )

        # VP7 预测
        vp7_model_dir = os.path.join(config["save_dir"], "VP7", f"fold{fold}", config["best_model_path"])
        vp7_probs = predict_probs(
            df["VP7_Seq"].tolist(), vp7_tokenizer, config["max_length"]["VP7"],
            vp7_model_dir, num_labels=len(label2idx)
        )

        # 平均两个模型的预测概率
        mean_probs = (vp4_probs + vp7_probs) / 2
        preds = np.argmax(mean_probs, axis=1)
        all_preds.append(preds)

        # 报告
        f1 = f1_score(all_labels, preds, average="macro")
        print(f"Fold {fold} Macro F1: {f1:.4f}")
        print("Confusion Matrix:")
        print(confusion_matrix(all_labels, preds))
        print("Classification Report:")
        print(classification_report(
            all_labels, preds,
            labels=list(label2idx.values()),  # 明确所有标签
            target_names=label_list,          # 标签名按顺序对应
            zero_division=0                   # 避免报错除0
        ))

    # 所有fold预测平均F1
    all_f1 = [f1_score(all_labels, p, average="macro") for p in all_preds]
    print(f"\n[Final] Mean F1: {np.mean(all_f1):.4f} ± {np.std(all_f1):.4f}")

if __name__ == "__main__":
    ensemble_test()
