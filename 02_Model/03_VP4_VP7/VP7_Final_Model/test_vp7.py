#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_vp4.py
读取 Rota_Info.filtered.typed.csv，对 VP4_aa 做预测，追加 pred_prob/pred_label，
并导出每个 Sample_ID 的 penultimate embedding（attention pooled 表征）到 .npy。

依赖：torch, transformers, pandas, numpy
工程依赖：model_utils.load_model, config.config
"""

import os
import sys
import argparse
import re
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import EsmTokenizer, AutoTokenizer

# 项目内模块
from config import config
from model_utils import load_model


def pick_tokenizer_name(user_tok: Optional[str]) -> str:
    """
    优先顺序：
    1) --tokenizer_name（若给且存在）
    2) config["model_name"]（若里面含有 tokenizer files）
    3) 常用的 ESM 模型（你 predict.py 里用的 8M）
    """
    candidates = []
    if user_tok:
        candidates.append(user_tok)
    if isinstance(config, dict) and "model_name" in config:
        candidates.append(config["model_name"])
    # 兜底到官方小模型（或你工程里常用路径）
    candidates.append("/media/server/DATA/lijiajun/project/Rotavirus/02_Model/esm_model/esm2_t6_8M_UR50D")
    candidates.append("facebook/esm2_t6_8M_UR50D")

    for name in candidates:
        try:
            _ = EsmTokenizer.from_pretrained(name)
            return name
        except Exception:
            try:
                _ = AutoTokenizer.from_pretrained(name)
                return name
            except Exception:
                continue
    raise RuntimeError("未能找到可用的 tokenizer，请显式传入 --tokenizer_name")


class SimpleSeqDataset(Dataset):
    """从 DataFrame 提供 (Sample_ID, 序列) 并做 tokenizer 编码"""
    def __init__(self, df: pd.DataFrame, id_col: str, seq_col: str, tokenizer, max_length: int):
        if id_col not in df.columns:
            raise ValueError(f"CSV 缺少列 {id_col}")
        if seq_col not in df.columns:
            raise ValueError(f"CSV 缺少列 {seq_col}")
        df = df.copy()
        df = df[~df[seq_col].isna()]
        self.ids = df[id_col].astype(str).tolist()
        # 清洗序列（去空白）
        self.seqs = [str(s).strip() for s in df[seq_col].astype(str).tolist()]
        self.tok = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        sid = self.ids[idx]
        seq = self.seqs[idx]
        enc = self.tok(
            seq,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        # 兼容性：无标签也给个占位
        item["labels"] = torch.tensor(0, dtype=torch.long)
        item["_sid"] = sid
        return item


def load_state_dict_flex(model: nn.Module, weight_path: str):
    """稳健加载权重：支持纯 state_dict 或 {state_dict: ...} 包装；自动剥离 'module.' 前缀"""
    state = torch.load(weight_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise RuntimeError("权重文件不是有效的 state_dict 字典")

    new_state = {}
    for k, v in state.items():
        nk = k[7:] if k.startswith("module.") else k
        new_state[nk] = v

    missing, unexpected = model.load_state_dict(new_state, strict=False)
    if missing:
        print("[WARN] Missing keys:", missing)
    if unexpected:
        print("[WARN] Unexpected keys:", unexpected)


def find_checkpoint_path(ckpt: Optional[str]) -> str:
    """
    若显式提供文件路径则直接用；
    若提供目录则在其中找 pytorch_model.bin；
    若未提供，则尝试 config 里的 best_model 路径（尽量少猜，还是建议显式传 --checkpoint）。
    """
    if ckpt:
        if os.path.isdir(ckpt):
            guess = os.path.join(ckpt, "pytorch_model.bin")
            if not os.path.exists(guess):
                raise FileNotFoundError(f"--checkpoint 是目录，但未找到 {guess}")
            return guess
        if os.path.isfile(ckpt):
            return ckpt
        raise FileNotFoundError(f"未找到权重文件：{ckpt}")

    # 尝试从 config 猜（如果你有固定存放结构，可在此拓展）
    default_guess = os.path.join(config.get("best_model_path", "best_model"), "pytorch_model.bin")
    if os.path.exists(default_guess):
        return default_guess
    raise FileNotFoundError("未提供 --checkpoint，且默认位置未找到权重。请显式指定。")


def register_penultimate_hook(model: nn.Module):
    """
    在“最后一个 nn.Linear”（即 ESMAttentionClassifier.classifier）上挂 pre_hook，
    抓取其输入作为 penultimate（attention pooled + dropout 之后）。
    """
    last_linear = None
    for m in model.modules():
        if isinstance(m, nn.Linear):
            last_linear = m
    if last_linear is None:
        return None, None

    cache = {"x": None}

    def pre_hook(module, inputs):
        cache["x"] = inputs[0].detach()

    handle = last_linear.register_forward_pre_hook(pre_hook)

    def getter():
        return cache["x"]

    return handle, getter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", type=str, required=True, help="Rota_Info.filtered.typed.csv")
    ap.add_argument("--out_dir", type=str, default="out_test_vp4")
    ap.add_argument("--id_col", type=str, default="Sample_ID")
    ap.add_argument("--seq_col", type=str, default="VP4_aa", help="默认用 VP4_aa 做预测")
    ap.add_argument("--checkpoint", type=str, default=None, help="权重路径或包含 pytorch_model.bin 的目录")
    ap.add_argument("--num_labels", type=int, default=2, help="分类数：二分类=2；若为1则用sigmoid")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--max_length", type=int, default=None, help="默认取 config['max_length']['VP4']")
    ap.add_argument("--tokenizer_name", type=str, default=None, help="可选：覆盖默认 tokenizer")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    emb_dir = os.path.join(args.out_dir, "embeddings")
    os.makedirs(emb_dir, exist_ok=True)

    # ---- 读取输入表
    df = pd.read_csv(args.in_csv)
    if args.id_col not in df.columns:
        raise ValueError(f"CSV 缺少列 {args.id_col}")
    if args.seq_col not in df.columns:
        raise ValueError(f"CSV 缺少列 {args.seq_col}")

    # ---- tokenizer / max_len
    tok_name = pick_tokenizer_name(args.tokenizer_name)
    try:
        tokenizer = EsmTokenizer.from_pretrained(tok_name)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(tok_name)

    if args.max_length is None:
        # 默认用 VP4 的 max_length
        args.max_length = int(config.get("max_length", {}).get("VP4", 850))

    # ---- 模型与权重
    model = load_model(model_dir_or_name=None, num_labels=args.num_labels)  # 结构与训练一致
    ckpt_path = find_checkpoint_path(args.checkpoint)
    load_state_dict_flex(model, ckpt_path)
    print(f"[OK] 加载权重：{ckpt_path}")

    device = torch.device("cuda" if (args.device in ["auto", "cuda"] and torch.cuda.is_available()) else "cpu")
    model.to(device)
    model.eval()

    # ---- DataLoader
    ds = SimpleSeqDataset(df, id_col=args.id_col, seq_col=args.seq_col,
                          tokenizer=tokenizer, max_length=args.max_length)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    # ---- 注册 penultimate hook
    hook_handle, get_penultimate = register_penultimate_hook(model)
    export_features = get_penultimate is not None

    all_ids: List[str] = []
    all_probs: List[float] = []
    all_preds: List[int] = []
    per_class_probs: List[np.ndarray] = []
    emb_index_rows = []

    sigmoid = nn.Sigmoid()
    softmax = nn.Softmax(dim=-1)

    with torch.no_grad():
        for batch in dl:
            sample_ids = batch["_sid"]
            toks = {k: v.to(device) for k, v in batch.items() if k not in ["_sid"]}

            out = model(**toks)          # dict，包含 logits
            logits = out["logits"] if isinstance(out, dict) else out
            if logits.ndim == 1:
                logits = logits.unsqueeze(-1)

            # 概率与预测
            if args.num_labels == 1 or logits.size(-1) == 1:
                probs = sigmoid(logits.view(-1))
                preds = (probs > 0.5).long()
                probs_np = probs.detach().cpu().numpy()
                preds_np = preds.detach().cpu().numpy()
                # 存一个两列形式以便统一下游（p0=1-p, p1=p）
                probs_2c = torch.stack([1 - probs, probs], dim=-1).cpu().numpy()
            else:
                probs_all = softmax(logits)
                probs_np = probs_all[:, 1].detach().cpu().numpy()  # 第2列作为“正类概率”
                preds_np = probs_all.argmax(dim=-1).detach().cpu().numpy()
                probs_2c = probs_all.detach().cpu().numpy()

            all_ids.extend(list(sample_ids))
            all_probs.extend(list(probs_np))
            all_preds.extend(list(preds_np))
            per_class_probs.append(probs_2c)

            # 导出 embedding
            if export_features:
                pen = get_penultimate()
                if pen is not None:
                    pen = pen.detach().cpu()
                    # 若出现 [B, L, H]，做 mean 池化
                    if pen.ndim == 3:
                        pen = pen.mean(dim=1)
                    for sid, vec in zip(sample_ids, pen):
                        safe_sid = re.sub(r"[^A-Za-z0-9_.\-]", "_", str(sid))
                        path = os.path.join(emb_dir, f"{safe_sid}.npy")
                        np.save(path, vec.numpy())
                        emb_index_rows.append({
                            "Sample_ID": sid,
                            "embedding_path": path,
                            "dim": int(vec.numel()),
                        })

    # ---- 汇总输出表：在原表上 merge（按 Sample_ID）
    pred_df = pd.DataFrame({
        "Sample_ID": all_ids,
        "pred_prob": all_probs,
        "pred_label": all_preds,
    })
    out_df = df.merge(pred_df, on="Sample_ID", how="left")

    # 若是多分类，也写出每类概率列
    probs_all_np = np.concatenate(per_class_probs, axis=0) if per_class_probs else None
    if probs_all_np is not None and probs_all_np.shape[1] > 2:
        # 为每一类命名 vp4_prob_c{i}
        for i in range(probs_all_np.shape[1]):
            out_df[f"vp4_prob_c{i}"] = probs_all_np[:, i]

    csv_out = os.path.join(args.out_dir, "Rota_Info.filtered.typed.with_pred.csv")
    out_df.to_csv(csv_out, index=False)
    print(f"[OK] 写出：{csv_out}（{len(out_df)} 行）")

    if emb_index_rows:
        emb_idx = pd.DataFrame(emb_index_rows)
        emb_idx_out = os.path.join(args.out_dir, "embeddings_index.tsv")
        emb_idx.to_csv(emb_idx_out, sep="\t", index=False)
        print(f"[OK] 写出：{emb_idx_out}（{len(emb_idx)} 条）")
    else:
        print("[WARN] 未导出任何 embedding —— 可能 hook 未捕获到 penultimate（请确认模型最后一层为 nn.Linear）。")

    if hook_handle is not None:
        hook_handle.remove()


if __name__ == "__main__":
    main()
'''
python test_vp7.py \
  --in_csv Rota_Info.filtered.typed.csv \
  --out_dir out_test_vp7 \
  --checkpoint /media/server/DATA/lijiajun/project/Rotavirus/02_Model/03_VP4_VP7/VP7_Final_Model/output_8M_pretraining/VP7/final_run/best_model/pytorch_model.bin \
  --id_col Sample_ID \
  --seq_col VP7_aa \
  --num_labels 5 \
  --batch_size 16 \
  --device auto
'''