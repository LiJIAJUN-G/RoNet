# -*- coding: utf-8 -*-

'''
python test_infer_and_embed.py \
  --csv Rota_Info.filtered.csv \
  --best_model_dir VP4_Final_Model/output_8M_pretraining/VP4/final_run/best_model \
  --out_prefix outputs/VP4_test/VP4 \
  --target VP4 \
  --id_col Sample_ID \
  --batch_size 32

'''
import os, sys, json, argparse, logging
from typing import Optional, List, Dict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import EsmTokenizer

# 项目内模块：与训练保持同构
from config import config                               # 读取 model_name / max_length / batch_size / device  :contentReference[oaicite:2]{index=2}
from model_utils import load_model                      # 你的构造+加载函数（ESM + attention + classifier）

# ---------- 推理数据集（不强制有标签） ----------
class InferenceSeqDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer: EsmTokenizer, max_len: int,
                 seq_col: str = "Seq", id_col: Optional[str] = None):
        self.df = df.reset_index(drop=True)
        if seq_col not in self.df.columns:
            raise ValueError(f"缺少序列列 '{seq_col}'，现有列：{self.df.columns.tolist()}")
        self.seq_col = seq_col
        self.id_col = id_col if (id_col and id_col in self.df.columns) else None
        self.has_host = "Host" in self.df.columns

        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        seq = str(row[self.seq_col])
        toks = self.tokenizer(seq, return_tensors="pt", padding="max_length",
                              truncation=True, max_length=self.max_len)
        toks = {k: v.squeeze(0) for k, v in toks.items()}
        sample_id = str(row[self.id_col]) if self.id_col else str(idx)
        out = {"input_ids": toks["input_ids"],
               "attention_mask": toks["attention_mask"],
               "id": sample_id}
        if self.has_host:
            out["host"] = str(row["Host"])
        return out

# ---------- 工具 ----------
def build_idx2label_from_csv(df: pd.DataFrame) -> List[str]:
    """仅当 CSV 含 Host 列时，用排序后的唯一值构造标签名列表（可能与训练顺序不同，仅供参考）。"""
    if "Host" in df.columns:
        return sorted(df["Host"].dropna().astype(str).unique().tolist())
    return []

def main():
    ap = argparse.ArgumentParser("Test: load best model, export embeddings & predictions")
    ap.add_argument("--csv", required=True, help="测试集 CSV（需含 Seq 列，可选 Host / Sample_ID）")
    ap.add_argument("--best_model_dir", required=True, help="包含 pytorch_model.bin 的目录")
    ap.add_argument("--out_prefix", required=True, help="输出前缀（不含扩展名）")
    ap.add_argument("--target", choices=["VP4","VP7"], default="VP4", help="读取 config 中该 target 的长度/批大小")
    ap.add_argument("--seq_col", default="Seq", help="序列列名（默认 Seq）")
    ap.add_argument("--id_col", default=None, help="样本 ID 列名（默认用行号）")
    ap.add_argument("--batch_size", type=int, default=None, help="覆盖 config 的 batch_size")
    ap.add_argument("--device", default=None, help="cuda/cpu；默认用 config['device']")
    ap.add_argument("--tokenizer", default=None, help="覆盖 config['model_name'] 的分词器路径/名称")
    args = ap.parse_args()

    # 设备
    device = args.device or config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(os.path.dirname(args.out_prefix), exist_ok=True)

    # 日志
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    logging.info("=== Inference start ===")
    logging.info(f"CSV={args.csv}")
    logging.info(f"ModelDir={args.best_model_dir}")
    logging.info(f"OutPrefix={args.out_prefix}")
    logging.info(f"Target={args.target}  Device={device}")

    # 读取 CSV
    df = pd.read_csv(args.csv)

    # 分词器 / 长度
    tok_name = args.tokenizer or config["model_name"]  # 与训练一致（本地路径或HF名）  :contentReference[oaicite:3]{index=3}
    tokenizer = EsmTokenizer.from_pretrained(tok_name)
    try:
        max_len = int(config["max_length"][args.target])        # 例如 VP4: 850
    except KeyError:
        raise KeyError(f"config['max_length'] 缺少 {args.target} 配置，请在 config 中补充。")  # :contentReference[oaicite:4]{index=4}
    bs = int(args.batch_size or config["batch_size"].get(args.target, 32))  # 若缺省则用 32 兜底  :contentReference[oaicite:5]{index=5}

    # 数据集/加载器
    ds = InferenceSeqDataset(df, tokenizer, max_len, seq_col=args.seq_col, id_col=args.id_col)
    dl = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=0)

    # 构建并加载最优模型
    # num_labels 先猜测（若 CSV 有 Host 就用该唯一数；否则先给 2），实际以探测 logits 维度为准
    guess_num_labels = len(build_idx2label_from_csv(df)) or 2
    model = load_model(args.best_model_dir, num_labels=guess_num_labels)  # 与训练同构
    model.to(device).eval()

    # 探测分类器真实维度
    with torch.no_grad():
        probe = next(iter(dl))
        logits_probe = model(input_ids=probe["input_ids"].to(device),
                             attention_mask=probe["attention_mask"].to(device))["logits"]
        num_labels = int(logits_probe.shape[1])
    logging.info(f"Detected classifier size = {num_labels}")

    # 若 CSV 有 Host，给一个 idx->label 名称（注意：顺序未必等于训练时的顺序）
    idx2label_csv = build_idx2label_from_csv(df)
    if idx2label_csv and len(idx2label_csv) != num_labels:
        logging.warning("CSV 推断的标签数与模型分类器大小不一致，将仅输出 pred_idx，不输出 pred_label。")
        idx2label_csv = []

    # 注册 hook 捕获 classifier 的输入（attention pooling 后向量）
    pooled_batches = []
    def hook_pre(module, inp):
        pooled_batches.append(inp[0].detach().cpu().numpy())
    handle = model.classifier.register_forward_pre_hook(lambda m, i: hook_pre(m, i))

    # 推理
    all_ids, all_hosts, all_logits = [], [], []
    with torch.no_grad():
        for b in dl:
            ids = list(b["id"])
            attn = b["attention_mask"].to(device)
            inp = b["input_ids"].to(device)

            out = model(input_ids=inp, attention_mask=attn)
            logits = out["logits"].detach().cpu().numpy()

            all_ids.extend(ids)
            all_logits.append(logits)
            if "host" in b:
                all_hosts.extend(list(b["host"]))
    handle.remove()

    all_logits = np.concatenate(all_logits, axis=0)
    embeddings = np.concatenate(pooled_batches, axis=0)  # [N, D]
    probs = torch.softmax(torch.tensor(all_logits), dim=1).numpy()
    pred_idx = probs.argmax(axis=1)
    pred_label = [ (idx2label_csv[i] if idx2label_csv and i < len(idx2label_csv) else f"class_{i}") for i in pred_idx ]

    # 保存 embedding
    emb_path = f"{args.out_prefix}.embeddings.npy"
    np.save(emb_path, embeddings)

    # 保存预测 TSV
    out = pd.DataFrame({
        "id": all_ids,
        "pred_idx": pred_idx,
        "pred_label": pred_label
    })
    if len(all_hosts) == len(all_ids):
        out["Host"] = all_hosts  # 真实标签（若 CSV 提供）

    # 追加 logits/probs 列
    for i in range(all_logits.shape[1]):
        out[f"logit_{i}"] = all_logits[:, i]
    for i in range(probs.shape[1]):
        out[f"prob_{i}"] = probs[:, i]

    pred_path = f"{args.out_prefix}.pred.tsv"
    out.to_csv(pred_path, sep="\t", index=False)

    # 元信息
    meta = {
        "csv": os.path.abspath(args.csv),
        "best_model_dir": os.path.abspath(args.best_model_dir),
        "out_prefix": os.path.abspath(args.out_prefix),
        "target": args.target,
        "tokenizer": tok_name,
        "num_samples": len(all_ids),
        "classifier_size": num_labels,
        "embedding_path": os.path.abspath(emb_path),
        "pred_tsv_path": os.path.abspath(pred_path),
        "seq_col": args.seq_col,
        "id_col": args.id_col,
        "config_device": config.get("device", None),         # :contentReference[oaicite:6]{index=6}
        "config_max_length": config["max_length"].get(args.target, None),  # :contentReference[oaicite:7]{index=7}
        "config_batch_size": config["batch_size"].get(args.target, None),  # :contentReference[oaicite:8]{index=8}
    }
    with open(f"{args.out_prefix}.meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    logging.info(f"✅ Embeddings saved: {emb_path}")
    logging.info(f"✅ Predictions saved: {pred_path}")
    logging.info("=== Inference done ===")

if __name__ == "__main__":
    main()
