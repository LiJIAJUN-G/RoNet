import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import importlib.util
from transformers import EsmTokenizer
from sklearn.metrics import f1_score, matthews_corrcoef, roc_auc_score, average_precision_score
from sklearn.preprocessing import label_binarize


def load_module_from_path(mod_name, file_path):
    spec = importlib.util.spec_from_file_location(mod_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module {mod_name} from {file_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_pkg(root_dir, tag):
    cfg_path = os.path.join(root_dir, "config.py")
    mu_path = os.path.join(root_dir, "model_utils.py")

    cfg_mod = load_module_from_path(f"config_{tag}", cfg_path)

    # 临时注册 config 供 model_utils 内的相对导入使用
    sys.modules['config'] = cfg_mod
    try:
        mu_mod = load_module_from_path(f"model_utils_{tag}", mu_path)
    finally:
        sys.modules.pop('config', None)

    return cfg_mod.config, mu_mod.load_model


def load_tokenizer(cfg, override_path=None):
    for cand in [override_path, cfg.get("tokenizer_name"), cfg.get("model_name")]:
        if not cand:
            continue
        try:
            return EsmTokenizer.from_pretrained(cand)
        except Exception:
            continue
    raise RuntimeError("Tokenizer loading failed. Please pass --tokenizer_vp4 / --tokenizer_vp7.")


def load_classes(target):
    df = pd.read_csv(f"../00_Data/{target}/train.csv")
    return sorted(df["Host"].unique().tolist())


def safe_col(name: str) -> str:
    s = str(name)
    for ch in [' ', '/', '\\', '\t', '\n', '\r', '(', ')', '[', ']', '{', '}', ':', ';', ',']:
        s = s.replace(ch, '_')
    while '__' in s:
        s = s.replace('__', '_')
    return s.strip('_')


@torch.no_grad()
def predict_probs(seqs, tokenizer, model, max_len, device, batch_size=16):
    # 取最后一个 Linear 的 out_features 作为类别数
    head = None
    for m in model.modules():
        if isinstance(m, torch.nn.Linear):
            head = m
    if head is None:
        raise RuntimeError("No linear classifier head found.")
    C = head.out_features

    N = len(seqs)
    out = np.full((N, C), np.nan, dtype=np.float32)
    valid = [i for i, s in enumerate(seqs) if isinstance(s, str) and len(s.strip()) > 0]
    if not valid:
        return out

    for start in range(0, len(valid), batch_size):
        idxs = valid[start:start+batch_size]
        batch = [seqs[i] for i in idxs]
        enc = tokenizer(batch, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        logits = model(input_ids=input_ids, attention_mask=attention_mask)["logits"]
        prob = torch.softmax(logits, dim=-1).cpu().numpy()
        for p, i in enumerate(idxs):
            out[i] = prob[p]
    return out

def compute_metrics_block(df, probs, classes, label_col):
    """
    更稳健：逐类检查是否有正负样本；只对有效类别计算 AUC/PR-AUC，再宏平均。
    其余与原逻辑一致。
    """
    if label_col is None or probs is None:
        return None

    C = len(classes)
    idx2label = {i: lab for i, lab in enumerate(classes)}
    label2idx = {lab: i for i, lab in idx2label.items()}

    n = len(df)
    y_raw = df[label_col].values
    keep, y_true = [], []
    for i in range(n):
        row = probs[i] if probs is not None else None
        if row is None or np.isnan(row).all():
            continue
        if pd.isna(y_raw[i]):
            continue

        if isinstance(y_raw[i], str):
            if y_raw[i] not in label2idx:
                continue
            yi = label2idx[y_raw[i]]
        else:
            try:
                yi = int(y_raw[i])
            except Exception:
                continue
            if yi not in idx2label:
                continue

        keep.append(i)
        y_true.append(yi)

    if not keep:
        return None

    y_true = np.array(y_true, dtype=int)
    y_score = probs[keep]
    y_pred = np.nanargmax(y_score, axis=1)

    out = {
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "num_samples": len(keep),
    }

    # ---- 稳健 ROC-AUC / PR-AUC ----
    if C >= 2:
        roc_list, pr_list = [], []
        for c in range(C):
            y_c = (y_true == c).astype(int)   # 一对多标签
            s_c = y_score[:, c]

            pos = (y_c == 1).sum()
            neg = (y_c == 0).sum()
            if pos == 0 or neg == 0:
                # 该类没有同时拥有正负样本 -> 跳过
                continue
            if np.isnan(s_c).all():
                continue

            try:
                roc_list.append(roc_auc_score(y_c, s_c))
            except Exception:
                pass
            try:
                pr_list.append(average_precision_score(y_c, s_c))
            except Exception:
                pass

        out["roc_auc_macro"] = float(np.mean(roc_list)) if roc_list else np.nan
        out["pr_auc_macro"]  = float(np.mean(pr_list))  if pr_list  else np.nan
    else:
        out["roc_auc_macro"] = np.nan
        out["pr_auc_macro"]  = np.nan

    return out



def main(args):
    # 分别加载两套包
    cfg4, load_model4 = load_pkg(args.vp4_root, tag="vp4")
    cfg7, load_model7 = load_pkg(args.vp7_root, tag="vp7")

    # 各自 tokenizer
    tok4 = load_tokenizer(cfg4, args.tokenizer_vp4)
    tok7 = load_tokenizer(cfg7, args.tokenizer_vp7)

    device = torch.device(cfg4.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

    # 类别（与训练一致，决定分类头大小与列命名）
    classes4 = load_classes("VP4")
    classes7 = load_classes("VP7")

    # 各自权重目录
    vp4_model_dir = os.path.join(args.vp4_root, "output_8M_pretraining", "VP4", "final_run", cfg4["best_model_path"])
    vp7_model_dir = os.path.join(args.vp7_root, "output_8M_pretraining", "VP7", "final_run", cfg7["best_model_path"])

    # 正确 num_labels 构建并加载
    model4 = load_model4(vp4_model_dir, num_labels=len(classes4)).to(device).eval()
    model7 = load_model7(vp7_model_dir, num_labels=len(classes7)).to(device).eval()

    # 数据
    df = pd.read_csv(args.input_csv)
    n = len(df)
    has_vp4 = "VP4_aa" in df.columns
    has_vp7 = "VP7_aa" in df.columns
    if not has_vp4 and not has_vp7:
        raise ValueError("test.csv 必须至少包含 VP4_aa 或 VP7_aa 列")

    # 预测
    vp4_probs = None
    vp7_probs = None

    if has_vp4:
        vp4_probs = predict_probs(
            df["VP4_aa"].tolist(),
            tok4, model4, cfg4["max_length"]["VP4"], device
        )
        # 用 VP4 的类别名做列名
        for c, lab in enumerate(classes4):
            df[f"vp4_prob_{safe_col(lab)}"] = vp4_probs[:, c]

    if has_vp7:
        vp7_probs = predict_probs(
            df["VP7_aa"].tolist(),
            tok7, model7, cfg7["max_length"]["VP7"], device
        )
        # 用 VP7 的类别名做列名
        for c, lab in enumerate(classes7):
            df[f"vp7_prob_{safe_col(lab)}"] = vp7_probs[:, c]

    # 平均（需要类别一致）
    if has_vp4 and has_vp7 and classes4 != classes7:
        raise ValueError("VP4 与 VP7 的类别集合不一致，无法平均概率。")

    classes = classes4 if has_vp4 else classes7
    C = len(classes)
    idx2label = {i: lab for i, lab in enumerate(classes)}

    # 平均概率（按类别位对齐）
    avg_probs = np.full((n, C), np.nan, dtype=np.float32)
    for i in range(n):
        p4 = vp4_probs[i] if vp4_probs is not None else None
        p7 = vp7_probs[i] if vp7_probs is not None else None
        v4 = p4 is not None and not np.isnan(p4).all()
        v7 = p7 is not None and not np.isnan(p7).all()
        if v4 and v7:
            avg_probs[i] = (p4 + p7) / 2.0
        elif v4:
            avg_probs[i] = p4
        elif v7:
            avg_probs[i] = p7
        else:
            avg_probs[i] = np.full(C, np.nan, dtype=np.float32)

    # 用原始类别名命名 avg 概率列
    for c, lab in enumerate(classes):
        df[f"avg_prob_{safe_col(lab)}"] = avg_probs[:, c]

    # 预测标签（基于平均）
    pred_idx = [int(np.nanargmax(row)) if not np.isnan(row).all() else np.nan for row in avg_probs]
    df["avg_pred_idx"] = pred_idx
    df["avg_pred_label"] = df["avg_pred_idx"].apply(lambda x: idx2label[int(x)] if pd.notna(x) else np.nan)

    # ===== 指标：VP4 / VP7 / Avg 各自一行 =====
    label_col = "Host" if "Host" in df.columns else ("label" if "label" in df.columns else None)
    metrics_rows = []

    # VP4 单独
    if has_vp4 and vp4_probs is not None:
        m_vp4 = compute_metrics_block(df, vp4_probs, classes, label_col)
        if m_vp4 is not None:
            m_vp4["model"] = "vp4"
            metrics_rows.append(m_vp4)

    # VP7 单独
    if has_vp7 and vp7_probs is not None:
        m_vp7 = compute_metrics_block(df, vp7_probs, classes, label_col)
        if m_vp7 is not None:
            m_vp7["model"] = "vp7"
            metrics_rows.append(m_vp7)

    # 平均
    m_avg = compute_metrics_block(df, avg_probs, classes, label_col)
    if m_avg is not None:
        m_avg["model"] = "avg"
        metrics_rows.append(m_avg)

    # 写 metrics.csv（如有任意一项可评估）
    if metrics_rows:
        metrics_df = pd.DataFrame(metrics_rows, columns=[
            "model", "macro_f1", "mcc", "roc_auc_macro", "pr_auc_macro", "num_samples"
        ])
        metrics_path = os.path.join(
            os.path.dirname(args.output_csv) or ".",
            f"{os.path.splitext(os.path.basename(args.output_csv))[0]}_metrics.csv"
        )
        metrics_df.to_csv(metrics_path, index=False, encoding='utf-8-sig')

    # 保存预测结果
    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    df.to_csv(args.output_csv, index=False)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("-i","--input_csv", type=str, default="./test.csv")
    p.add_argument("-o","--output_csv", type=str, default="./pred_with_probs.csv")
    p.add_argument("--vp4_root", type=str, default="./VP4_Final_Model")
    p.add_argument("--vp7_root", type=str, default="./VP7_Final_Model")
    p.add_argument("--tokenizer_vp4", type=str, default="/media/server/DATA/lijiajun/project/Rotavirus/02_Model/esm_model/esm2_t6_8M_UR50D", help="显式指定 VP4 的 tokenizer 目录")
    p.add_argument("--tokenizer_vp7", type=str, default="/media/server/DATA/lijiajun/project/Rotavirus/02_Model/esm_model/esm2_t6_8M_UR50D", help="显式指定 VP7 的 tokenizer 目录")
    args = p.parse_args()
    main(args)
