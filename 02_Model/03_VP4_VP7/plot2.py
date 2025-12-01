# -*- coding: utf-8 -*-
"""
从 test_probs.csv 生成三张图：
1) 宏平均 ROC 曲线（VP4 / VP7 / Average）+ 置信区间
2) 宏平均 PR 曲线（VP4 / VP7 / Average）+ 置信区间
3) Average 的归一化混淆矩阵（ratio 和 n/row_total 换行显示；删除真实样本数为0的类别）
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score,
    confusion_matrix
)
from sklearn.utils import resample

CSV = "test_probs.csv"

# ===== 样式 =====
mpl.rcParams["font.family"] = "Arial"
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42

def style_axes(ax, title):
    ax.set_title(title, fontsize=16)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)
    ax.tick_params(labelsize=12)
    ax.grid(False)

# ===== 读取 =====
df = pd.read_csv(CSV)

# ===== 提取类别 =====
avg_cols = [c for c in df.columns if c.startswith("avg_prob_")]
classes = [c.replace("avg_prob_", "") for c in avg_cols]

P4  = df[[f"vp4_prob_{c}" for c in classes]].to_numpy(float)
P7  = df[[f"vp7_prob_{c}" for c in classes]].to_numpy(float)
Pav = df[[f"avg_prob_{c}" for c in classes]].to_numpy(float)

label_to_idx = {c: i for i, c in enumerate(classes)}
y_true = df["Host"].astype(str).values
unknown = sorted(set(y for y in y_true if y not in label_to_idx))
if unknown:
    raise ValueError(f"发现未知类别（不在 avg_prob_* 列中）：{unknown}")
y_idx = np.array([label_to_idx[y] for y in y_true])
Y = np.zeros((len(y_idx), len(classes)), dtype=int)
Y[np.arange(len(y_idx)), y_idx] = 1

# ===== 宏平均 ROC / PR =====
def macro_roc(Y, P):
    mean_fpr = np.linspace(0, 1, 200)
    tpr_sum, aucs, valid = np.zeros_like(mean_fpr), [], 0
    for k in range(Y.shape[1]):
        yk = Y[:, k]
        if yk.sum() == 0 or yk.sum() == len(yk):
            continue
        fpr, tpr, _ = roc_curve(yk, P[:, k])
        tpr_sum += np.interp(mean_fpr, fpr, tpr)
        aucs.append(auc(fpr, tpr))
        valid += 1
    return mean_fpr, tpr_sum / valid, np.mean(aucs)

def macro_pr(Y, P):
    grid_recall = np.linspace(0, 1, 200)
    prec_sum, aps, valid = np.zeros_like(grid_recall), [], 0
    for k in range(Y.shape[1]):
        yk = Y[:, k]
        if yk.sum() == 0 or yk.sum() == len(yk):
            continue
        precision, recall, _ = precision_recall_curve(yk, P[:, k])
        prec_sum += np.interp(grid_recall, recall[::-1], precision[::-1])
        aps.append(average_precision_score(yk, P[:, k]))
        valid += 1
    return grid_recall, prec_sum / valid, np.mean(aps)

# ===== Bootstrap 置信区间函数 =====
def bootstrap_ci(func, Y, P, n_boot=1000, alpha=0.05):
    curves = []
    n = len(Y)
    for _ in range(n_boot):
        idx = np.random.randint(0, n, n)
        try:
            x, y, _ = func(Y[idx], P[idx])
            curves.append(y)
        except Exception:
            continue
    curves = np.array(curves)
    lower = np.nanpercentile(curves, 100*alpha/2, axis=0)
    upper = np.nanpercentile(curves, 100*(1-alpha/2), axis=0)
    return lower, upper

# ===== 1) ROC =====
fig, ax = plt.subplots(figsize=(6.2, 5))
colors = ["#84C3B7", "#7DA6C6", "#EAAA60"]  # 蓝、红、绿
for (name, P, color) in [("VP4", P4, colors[0]), ("VP7", P7, colors[1]), ("Fusion", Pav, colors[2])]:
    fpr, tpr, macro_auc = macro_roc(Y, P)
    ax.plot(fpr, tpr, label=f"{name} (AUC={macro_auc:.3f})", color=color, linewidth=2.2)
    # 置信区间
    lower, upper = bootstrap_ci(macro_roc, Y, P, n_boot=500)
    ax.fill_between(fpr, lower, upper, color=color, alpha=0.2)
ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
ax.set_xlabel("False Positive Rate", fontsize=14)
ax.set_ylabel("True Positive Rate", fontsize=14)
style_axes(ax, "Macro-averaged ROC (Fusion, VP4-VP7)")
ax.legend(frameon=False, fontsize=13)
fig.tight_layout()
fig.savefig("roc_macro.png", dpi=300)
fig.savefig("roc_macro.pdf")

# ===== 2) PR =====
fig, ax = plt.subplots(figsize=(6.2, 5))
for (name, P, color) in [("VP4", P4, colors[0]), ("VP7", P7, colors[1]), ("Fusion", Pav, colors[2])]:
    recall, precision, macro_ap = macro_pr(Y, P)
    ax.plot(recall, precision, label=f"{name} (AP={macro_ap:.3f})", color=color, linewidth=2.2)
    # 置信区间
    lower, upper = bootstrap_ci(macro_pr, Y, P, n_boot=500)
    ax.fill_between(recall, lower, upper, color=color, alpha=0.2)
baseline = 1 / Y.shape[1]
ax.hlines(baseline, 0, 1, colors="gray", linestyles="--", linewidth=1, label=f"Baseline={baseline:.2f}")
ax.set_xlabel("Recall", fontsize=14)
ax.set_ylabel("Precision", fontsize=14)
style_axes(ax, "Macro-averaged PR (Fusion, VP4-VP7)")
ax.legend(frameon=False, fontsize=13)
fig.tight_layout()
fig.savefig("pr_macro.png", dpi=300)
fig.savefig("pr_macro.pdf")

# ===== 3) Average 混淆矩阵 =====
if "avg_pred_label" in df.columns:
    y_pred = df["avg_pred_label"].astype(str).values
else:
    y_pred = np.array([classes[i] for i in np.argmax(Pav, axis=1)])

cm_full = confusion_matrix(y_true, y_pred, labels=classes)
row_totals = cm_full.sum(axis=1)
keep_idx = [i for i, tot in enumerate(row_totals) if tot > 0]
keep_classes = [classes[i] for i in keep_idx]

cm = cm_full[np.ix_(keep_idx, keep_idx)]
row_totals_kept = row_totals[keep_idx]
cm_norm = cm.astype(float) / np.clip(row_totals_kept[:, None], 1, None)

fig, ax = plt.subplots(figsize=(6.6, 6.2))
im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
ax.set_xticks(np.arange(len(keep_classes)))
ax.set_yticks(np.arange(len(keep_classes)))
ax.set_xticklabels(keep_classes, fontsize=12, rotation=30, ha="right")
ax.set_yticklabels(keep_classes, fontsize=12)
style_axes(ax, "Fusion — Normalized Confusion Matrix")
ax.set_xlabel("Predicted label", fontsize=14)
ax.set_ylabel("True label", fontsize=14)

for i in range(len(keep_classes)):
    for j in range(len(keep_classes)):
        n = cm[i, j]
        tot = int(row_totals_kept[i])
        ratio = 0.0 if tot == 0 else cm_norm[i, j]
        ax.text(
            j, i,
            f"{ratio:.2f}\n({n}/{tot})",
            ha="center", va="center",
            fontsize=10,
            color=("white" if ratio > 0.5 else "black")
        )

cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("Proportion", fontsize=12)
fig.tight_layout()
fig.savefig("cm_avg.png", dpi=300)
fig.savefig("cm_avg.pdf")

print("[完成] 已保存: roc_macro.png/pdf, pr_macro.png/pdf, cm_avg.png/pdf")
