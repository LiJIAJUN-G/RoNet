# predict_host_probs.py
# -*- coding: utf-8 -*-

import os
import sys
import json
import argparse
import logging
import numpy as np
import pandas as pd
import torch

from transformers import EsmTokenizer, Trainer, TrainingArguments
from sklearn.metrics import (
    f1_score, confusion_matrix, classification_report,
    matthews_corrcoef, roc_auc_score, average_precision_score
)
from sklearn.preprocessing import label_binarize

# 你项目里的模块
from model_utils import load_model
from config import config
from dataset import HostDataset  # 可选使用

LOG = logging.getLogger("predict")
LOG.setLevel(logging.INFO)
LOG.addHandler(logging.StreamHandler(sys.stdout))

def build_label_maps(train_csv: str):
    df = pd.read_csv(train_csv)
    all_labels = sorted(df["Host"].astype(str).unique())
    label2idx = {lab: i for i, lab in enumerate(all_labels)}
    idx2label = {i: lab for lab, i in label2idx.items()}
    return label2idx, idx2label

class InferenceDataset(torch.utils.data.Dataset):
    """最小化推理数据集（无 Host 标签时使用）"""
    def __init__(self, df: pd.DataFrame, seq_col: str, tokenizer, max_len: int):
        if seq_col not in df.columns:
            raise ValueError(f"未在输入表中找到序列列：{seq_col}")
        self.ids = df.index.to_list()
        self.seqs = df[seq_col].astype(str).fillna("").tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, i):
        seq = self.seqs[i]
        enc = self.tokenizer(
            seq,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        # Trainer 需要 labels 键存在；无标签给个占位
        item["labels"] = torch.tensor(0, dtype=torch.long)
        return item

def main():
    ap = argparse.ArgumentParser("Predict host probabilities for VP4/VP7")
    ap.add_argument("--target", type=str, required=True, choices=["VP4", "VP7"])
    ap.add_argument("--in_csv", type=str, required=True, help="输入待预测CSV")
    ap.add_argument("--out_csv", type=str, required=True, help="输出带概率的CSV")
    ap.add_argument("--seq_col", type=str, default=None,
                    help="序列列名（默认：VP4_msa / VP7_msa）")
    ap.add_argument("--id_col", type=str, default="Sample_ID",
                    help="可选：样本ID列名（若存在会在输出中保留）")
    ap.add_argument("--use_hostdataset", action="store_true",
                    help="即使没有Host列也强制使用项目内HostDataset（若其支持无标签）")
    ap.add_argument("--model_weights", type=str, default=None,
                    help="pytorch_model.bin 路径；默认读取 config['save_dir']/target/final_run/best_model_path/")
    args = ap.parse_args()

    target = args.target
    save_dir = config["save_dir"]
    max_len = config["max_length"][target]

    tokenizer = EsmTokenizer.from_pretrained(
        "/media/server/DATA/lijiajun/project/Rotavirus/02_Model/esm_model/esm2_t6_8M_UR50D"
    )

    train_csv = f"../../00_Data/{target}/train.csv"
    test_csv  = f"../../00_Data/{target}/test.csv"  # 仅用于复用 HostDataset 的列名逻辑时参考
    label2idx, idx2label = build_label_maps(train_csv)
    num_labels = len(label2idx)

    # ---- 模型与权重 ----
    model = load_model(None, num_labels=num_labels)

    if args.model_weights is None:
        model_base = os.path.join(save_dir, target, "final_run", config["best_model_path"])
        args.model_weights = os.path.join(model_base, "pytorch_model.bin")

    if not os.path.exists(args.model_weights):
        raise FileNotFoundError(f"未找到模型权重：{args.model_weights}")
    state = torch.load(args.model_weights, map_location="cpu")
    model.load_state_dict(state)
    LOG.info(f"[OK] 加载权重：{args.model_weights}")

    # ---- 读取输入 ----
    df_in = pd.read_csv(args.in_csv)
    has_label = "Host" in df_in.columns

    # 默认序列列
    if args.seq_col is None:
        args.seq_col = f"{target}_msa"  # 常用
        if args.seq_col not in df_in.columns:
            # 回退尝试原始aa列
            fallback = f"{target}_aa"
            if fallback in df_in.columns:
                args.seq_col = fallback

    # ---- 构造dataset ----
    if has_label or args.use_hostdataset:
        # 若 HostDataset 在你的项目里依赖 Host 列，但这里无 Host 会报错，此时请使用 InferenceDataset
        ds = HostDataset(args.in_csv, tokenizer, max_len, label2idx)
        LOG.info(f"[DS] 使用 HostDataset（has_label={has_label}）")
    else:
        ds = InferenceDataset(df_in, args.seq_col, tokenizer, max_len)
        LOG.info(f"[DS] 使用 InferenceDataset（无 Host 标签）")

    # ---- 推理 ----
    tmp_out = os.path.join(save_dir, target, "predict_tmp")
    os.makedirs(tmp_out, exist_ok=True)
    targs = TrainingArguments(output_dir=tmp_out, per_device_eval_batch_size=32, dataloader_drop_last=False)
    trainer = Trainer(model=model, args=targs)

    pred = trainer.predict(ds)
    logits = pred.predictions
    probs = torch.softmax(torch.tensor(logits), dim=1).numpy()
    preds = np.argmax(probs, axis=1)

    # ---- 组装输出 ----
    out = df_in.copy()
    # 概率列名：vp4_prob_* / vp7_prob_*
    prefix = target.lower()
    for i, lab in idx2label.items():
        col = f"{prefix}_prob_{lab}"
        out[col] = probs[:, i]

    out["pred_host"] = [idx2label[i] for i in preds]
    out["pred_prob"] = probs[np.arange(len(preds)), preds]

    # ---- 如果有真实Host，计算指标 ----
    if has_label:
        y_true_labels = df_in["Host"].astype(str).tolist()
        # 将真实标签映射到训练标签空间；若遇到未见过的标签，先过滤
        valid_mask = [lab in label2idx for lab in y_true_labels]
        if not all(valid_mask):
            LOG.info(f"[WARN] 有 {sum(~np.array(valid_mask))} 条样本的真实Host不在训练标签集中，将在评估中忽略。")
        y_true_idx = np.array([label2idx[lab] for lab in y_true_labels if lab in label2idx])
        y_pred_idx = preds[np.array(valid_mask)]

        existing_labels = sorted(set(y_true_idx) | set(y_pred_idx))
        target_names = [idx2label[i] for i in existing_labels]

        macro_f1 = f1_score(y_true_idx, y_pred_idx, labels=existing_labels, average="macro")
        mcc = matthews_corrcoef(y_true_idx, y_pred_idx)

        # AUC
        y_true_bin = label_binarize(y_true_idx, classes=existing_labels)
        y_score_sub = probs[np.array(valid_mask)][:, existing_labels]
        try:
            roc_auc_macro = roc_auc_score(y_true_bin, y_score_sub, average="macro", multi_class="ovr")
        except Exception as e:
            roc_auc_macro = np.nan
            LOG.info(f"ROC-AUC 计算失败：{e}")
        try:
            pr_auc_macro = average_precision_score(y_true_bin, y_score_sub, average="macro")
        except Exception as e:
            pr_auc_macro = np.nan
            LOG.info(f"PR-AUC 计算失败：{e}")

        cm = confusion_matrix(y_true_idx, y_pred_idx, labels=existing_labels)
        report = classification_report(y_true_idx, y_pred_idx, labels=existing_labels, target_names=target_names)

        LOG.info(f"[{target}] Metrics on {args.in_csv}")
        LOG.info(f"Macro F1 : {macro_f1:.4f}")
        LOG.info(f"MCC      : {mcc:.4f}")
        LOG.info(f"ROC-AUC  : {roc_auc_macro if not np.isnan(roc_auc_macro) else 'NaN'}")
        LOG.info(f"PR-AUC   : {pr_auc_macro if not np.isnan(pr_auc_macro) else 'NaN'}")
        LOG.info("Confusion Matrix:\n" + str(cm))
        LOG.info("Classification Report:\n" + report)

    # ---- 保存 ----
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    LOG.info(f"[SAVED] {args.out_csv}")

if __name__ == "__main__":
    main()

'''
python predict.py \
  --target VP7 \
  --seq_col Seq \
  --in_csv ./test.csv \
  --out_csv ./VP7_test.with_probs.csv

'''