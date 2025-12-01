# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path

# ================== 配置 ==================
ROOTS = [
    "01_VP4_MLP",
    "01_VP4_Att",
    "01_VP4_Att_Pretraining",
]
CANDIDATE_DIRS = ["output_8M", "output_8M_pretraining"]
CSV_NAME = "test_f1_by_fold.csv"

METRICS = ["macro_f1", "mcc", "roc_auc_macro", "pr_auc_macro"]
DISPLAY_NAMES = {
    "01_VP4_MLP": "VP4-MLP",
    "01_VP4_Att": "VP4-Att",
    "01_VP4_Att_Pretraining": "VP4-Att (Pretrain)",
}

# 样式
mpl.rcParams["font.family"] = "Arial"
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
COLORS = ["#84C3B7", "#7DA6C6", "#E68B81"]

OUT_PNG = "vp4_fold_metrics_bar_scatter_error.png"
OUT_PDF = "vp4_fold_metrics_bar_scatter_error.pdf"

# ================== 读取函数 ==================
def find_csv(root: str) -> str | None:
    for sub in CANDIDATE_DIRS:
        p = Path(root) / sub / "VP4" / CSV_NAME
        if p.is_file():
            return str(p)
    return None

def read_folds(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    keep = df["fold"].astype(str).str.fullmatch(r"[1-5]")
    return df.loc[keep, ["fold"] + METRICS].assign(fold=lambda x: x["fold"].astype(int))

# ================== 收集数据 ==================
per_method = {}
present_roots = []
for i, root in enumerate(ROOTS):
    csv_path = find_csv(root)
    if not csv_path:
        print(f"[提示] 跳过，未找到结果CSV：{root}")
        continue
    try:
        df = read_folds(csv_path)
        for m in METRICS:
            df[m] = pd.to_numeric(df[m], errors="coerce")
        per_method[root] = {m: df[m].dropna().values for m in METRICS}
        present_roots.append(root)
        print(f"[OK] 读取：{csv_path}；有效折数：{len(df)}")
    except Exception as e:
        print(f"[跳过] 解析失败 {csv_path}: {e}")

if not present_roots:
    raise SystemExit("[错误] 未读取到任何方法的有效CSV。")

# ================== 统计 ==================
means = {root: [np.nanmean(per_method[root].get(m, np.array([np.nan]))) for m in METRICS]
         for root in present_roots}
stds = {root: [np.nanstd(per_method[root].get(m, np.array([np.nan])), ddof=1) for m in METRICS]
        for root in present_roots}

# ================== 绘图 ==================
plt.figure(figsize=(8,6))
x = np.arange(len(METRICS))
n_methods = len(present_roots)
bar_width = 0.8 / n_methods
np.random.seed(0)
scatter_jitter = bar_width * 0.25

for i, root in enumerate(present_roots):
    color = COLORS[i % len(COLORS)]
    offs = x - 0.4 + bar_width / 2 + i * bar_width
    # 柱形图（无黑边，误差线为灰色）
    bars = plt.bar(
        offs, means[root], yerr=stds[root], width=bar_width,
        capsize=3, label=DISPLAY_NAMES.get(root, root),
        color=color, edgecolor="none", ecolor="gray", linewidth=0.6
    )
    # 每折散点
    for j, metric in enumerate(METRICS):
        vals = per_method[root].get(metric, np.array([]))
        if vals.size == 0:
            continue
        xs = np.full(vals.shape, offs[j]) + (np.random.rand(vals.size) - 0.5) * 2 * scatter_jitter
        plt.scatter(xs, vals, s=24, facecolors="white", edgecolors=color, linewidths=1.5, zorder=3)

    # ==== 在 VP4-Att (Pretrain) 的柱子上标注数值 ====
    if root == "01_VP4_Att_Pretraining":
        for j, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2, height + 0.002,
                f"{height:.3f}", ha="center", va="bottom",
                fontsize=13, fontweight="bold", color="black"
            )


# 坐标样式
plt.xticks(x, ["macro-F1", "MCC", "ROC-AUC (macro)", "PR-AUC (macro)"], fontsize=14)
plt.ylabel("Value", fontsize=14)
plt.yticks(fontsize=14)
plt.ylim(0.8, 1.02)
plt.title("VP4 Models(5-Fold)", fontsize=14)
plt.legend(frameon=False, fontsize=12, ncol=min(3, n_methods))

# 去掉背景网格，只保留左下框线
plt.grid(False)
ax = plt.gca()
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
ax.spines["left"].set_linewidth(1.0)
ax.spines["bottom"].set_linewidth(1.0)

plt.tight_layout()
plt.savefig(OUT_PNG, dpi=300)
plt.savefig(OUT_PDF)
plt.show()

print(f"[完成] 已保存: {OUT_PNG}, {OUT_PDF}")
