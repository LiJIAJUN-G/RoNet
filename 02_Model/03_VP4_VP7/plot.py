# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# ========== 配置 ==========
CSV_FILE = "test_probs_metrics.csv"
METRICS = ["macro_f1", "mcc", "roc_auc_macro", "pr_auc_macro"]
DISPLAY_NAMES = {
    "vp4": "VP4",
    "vp7": "VP7",
    "avg": "Average"
}
COLORS = ["#4285F4", "#DB4437", "#F4B400"]  # Google风格：蓝/红/黄

mpl.rcParams["font.family"] = "Arial"
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42

OUT_PNG = "test_probs_metrics_bar.png"
OUT_PDF = "test_probs_metrics_bar.pdf"

# ========== 读取 ==========
df = pd.read_csv(CSV_FILE)
df = df.set_index("model")

# ========== 绘图 ==========
plt.figure(figsize=(9, 5))
x = np.arange(len(METRICS))  # 指标位置
n_models = len(df)
bar_width = 0.8 / n_models

for i, model in enumerate(df.index):
    values = df.loc[model, METRICS].values.astype(float)
    offs = x - 0.4 + bar_width/2 + i * bar_width
    plt.bar(
        offs, values, width=bar_width,
        label=DISPLAY_NAMES.get(model, model),
        color=COLORS[i % len(COLORS)], edgecolor="black", linewidth=0.6
    )
    # 柱顶加数值标签
    for j, v in enumerate(values):
        plt.text(offs[j], v + 0.005, f"{v:.3f}", ha="center", va="bottom", fontsize=9)

plt.xticks(x, ["macro-F1", "MCC", "ROC-AUC (macro)", "PR-AUC (macro)"], fontsize=12)
plt.ylabel("Score", fontsize=13)
plt.ylim(0.8, 1.05)
plt.title("Test Probs Metrics by Model", fontsize=15)
plt.legend(frameon=False, fontsize=11)
plt.tight_layout()
plt.savefig(OUT_PNG, dpi=300)
plt.savefig(OUT_PDF)
plt.show()

print(f"[完成] 已保存: {OUT_PNG}, {OUT_PDF}")
