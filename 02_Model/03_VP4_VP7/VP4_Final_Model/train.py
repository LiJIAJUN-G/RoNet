import os
import torch
import numpy as np
import pandas as pd
import logging
from transformers import (
    EsmTokenizer,
    Trainer,
    TrainingArguments,
)
from sklearn.metrics import (
    f1_score, confusion_matrix, classification_report,
    matthews_corrcoef, roc_auc_score, average_precision_score
)
from sklearn.preprocessing import label_binarize
from dataset import HostDataset
from model_utils import load_model
from config import config

def train_for_target(target):
    # ✅ 日志
    log_path = os.path.join(config["save_dir"], target)
    os.makedirs(log_path, exist_ok=True)
    log_file = os.path.join(log_path, "train.log")

    logger = logging.getLogger(f"logger_{target}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(log_file); fh.setLevel(logging.INFO)
    ch = logging.StreamHandler();       ch.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(fmt); ch.setFormatter(fmt)
    logger.addHandler(fh); logger.addHandler(ch)

    logger.info(f"========== Start Training for {target} (use ALL data, no folds) ==========")

    tokenizer = EsmTokenizer.from_pretrained("/media/server/DATA/lijiajun/project/Rotavirus/02_Model/esm_model/esm2_t6_8M_UR50D")
    max_len = config["max_length"][target]
    train_full_path = f"../../00_Data/{target}/train.csv"
    test_path = f"../../00_Data/{target}/test.csv"

    # 🔁 统一 label2idx
    train_full_df = pd.read_csv(train_full_path)
    all_labels = sorted(train_full_df["Host"].unique())
    label2idx = {label: i for i, label in enumerate(all_labels)}
    idx2label = {i: label for label, i in label2idx.items()}

    # === 用全部训练集构建数据集 ===
    train_ds = HostDataset(train_full_path, tokenizer, max_len, label2idx)

    # === 初始化模型 ===
    model = load_model(None, num_labels=len(label2idx))

    # === 训练参数：不做评估/不保存中间检查点，最后手动保存最终模型 ===
    output_dir = os.path.join(config['save_dir'], target, "final_run")
    os.makedirs(output_dir, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="no",
        save_strategy="no",
        load_best_model_at_end=False,
        per_device_train_batch_size=config["batch_size"][target],
        num_train_epochs=config["epochs"],
        learning_rate=config["lr"],
        logging_strategy="steps",
        logging_steps=50,
        report_to="tensorboard",
        logging_dir=os.path.join(output_dir, "tb_logs"),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
    )

    # === 训练（全量数据，无验证）===
    trainer.train()

    # === 保存“最后一轮”的模型权重 ===
    final_best_model_dir = os.path.join(output_dir, config["best_model_path"])
    os.makedirs(final_best_model_dir, exist_ok=True)
    torch.save(trainer.model.state_dict(), os.path.join(final_best_model_dir, "pytorch_model.bin"))
    logger.info(f"✅ 最后一轮模型权重已保存到 {final_best_model_dir}")

    # ========== 测试集评估（单次模型） ==========
    test_ds = HostDataset(test_path, tokenizer, max_len, label2idx)
    logger.info(f"\n[🔍 Test on {target}] Start testing final model on test.csv")

    pred_output = trainer.predict(test_ds)
    logits = pred_output.predictions
    labels = pred_output.label_ids
    preds = np.argmax(logits, axis=1)

    # 仅对测试中出现的类计算，防止缺项报错
    existing_labels = sorted(set(preds) | set(labels))
    target_names = [idx2label[i] for i in existing_labels]

    macro_f1 = f1_score(labels, preds, labels=existing_labels, average="macro")
    mcc = matthews_corrcoef(labels, preds)

    # 概率用于 AUC
    probs = torch.softmax(torch.tensor(logits), dim=1).numpy()
    if len(existing_labels) >= 2:
        y_true_bin = label_binarize(labels, classes=existing_labels)   # [N, C_present]
        y_score_sub = probs[:, existing_labels]                        # [N, C_present]
        try:
            roc_auc_macro = roc_auc_score(y_true_bin, y_score_sub, average="macro", multi_class="ovr")
        except Exception as e:
            roc_auc_macro = np.nan
            logger.info(f"ROC-AUC calculation skipped: {e}")
        try:
            pr_auc_macro = average_precision_score(y_true_bin, y_score_sub, average="macro")
        except Exception as e:
            pr_auc_macro = np.nan
            logger.info(f"PR-AUC calculation skipped: {e}")
    else:
        roc_auc_macro = np.nan
        pr_auc_macro = np.nan

    # 日志输出
    logger.info(f"Macro F1-score: {macro_f1:.4f}")
    logger.info(f"MCC: {mcc:.4f}")
    logger.info(f"ROC-AUC (macro, OvR): {roc_auc_macro if not np.isnan(roc_auc_macro) else 'NaN'}")
    logger.info(f"PR-AUC  (macro): {pr_auc_macro if not np.isnan(pr_auc_macro) else 'NaN'}")
    cm = confusion_matrix(labels, preds, labels=existing_labels)
    logger.info("Confusion Matrix:\n" + str(cm))
    report = classification_report(labels, preds, labels=existing_labels, target_names=target_names)
    logger.info("Classification Report:\n" + report)

    # === 保存 CSV（单行结果）===
    df_metrics = pd.DataFrame([{
        "split": "test",
        "macro_f1": macro_f1,
        "mcc": mcc,
        "roc_auc_macro": roc_auc_macro,
        "pr_auc_macro": pr_auc_macro
    }])
    result_csv_path = os.path.join(config["save_dir"], target, config["f1_csv"])
    df_metrics.to_csv(result_csv_path, index=False)
    logger.info(f"[Saved] Final test metrics saved to: {result_csv_path}")

if __name__ == "__main__":
    targets = ["VP4"]  # 或 ["VP7"], 或二者 ["VP4","VP7"]
    for target in targets:
        train_for_target(target)
