import os
import shutil
import torch
import numpy as np
import pandas as pd
import logging
from transformers import (
    EsmTokenizer,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
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
    # ✅ 设置日志
    log_path = os.path.join(config["save_dir"], target)
    os.makedirs(log_path, exist_ok=True)
    log_file = os.path.join(log_path, "train.log")

    logger = logging.getLogger(f"logger_{target}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    logger.info(f"========== Start Training for {target} ==========")

    tokenizer = EsmTokenizer.from_pretrained("/media/server/DATA/lijiajun/project/Rotavirus/02_Model/esm_model/esm2_t6_8M_UR50D")
    max_len = config["max_length"][target]
    fold_path = f"../00_Data/{target}/fold"
    test_path = f"../00_Data/{target}/test.csv"
    train_full_path = f"../00_Data/{target}/train.csv"
    best_f1 = 0.0

    # 🔁 提取全量 label2idx，保持训练和测试一致
    train_full_df = pd.read_csv(train_full_path)
    all_labels = sorted(train_full_df["Host"].unique())
    label2idx = {label: i for i, label in enumerate(all_labels)}
    idx2label = {i: label for label, i in label2idx.items()}

    # for fold in range(1, 6):
    #     logger.info(f"------ Fold {fold} Training ------")
    #     train_file = f"{fold_path}/fold{fold}_train.csv"
    #     val_file = f"{fold_path}/fold{fold}_val.csv"
    #     train_ds = HostDataset(train_file, tokenizer, max_len, label2idx)
    #     val_ds = HostDataset(val_file, tokenizer, max_len, label2idx)

    #     model = load_model(None, num_labels=len(label2idx))

    #     output_dir = os.path.join(config['save_dir'], target, f"fold{fold}")
    #     training_args = TrainingArguments(
    #         output_dir=output_dir,
    #         eval_strategy="epoch",
    #         save_strategy="epoch",
    #         save_total_limit=1,
    #         load_best_model_at_end=True,
    #         metric_for_best_model="eval_f1",
    #         greater_is_better=True,
    #         per_device_train_batch_size=config["batch_size"][target],
    #         per_device_eval_batch_size=config["batch_size"][target],
    #         num_train_epochs=config["epochs"],
    #         learning_rate=config["lr"],
    #         logging_strategy="epoch",
    #         logging_dir=os.path.join(output_dir, "tb_logs"),
    #         report_to="tensorboard",
    #     )

    #     def compute_metrics(eval_pred):
    #         logits, labels = eval_pred
    #         preds = np.argmax(logits, axis=1)
    #         macro_f1 = f1_score(labels, preds, average="macro")
    #         return {"f1": macro_f1}

    #     trainer = Trainer(
    #         model=model,
    #         args=training_args,
    #         train_dataset=train_ds,
    #         eval_dataset=val_ds,
    #         compute_metrics=compute_metrics,
    #         callbacks=[EarlyStoppingCallback(early_stopping_patience=config["early_stopping_patience"])]
    #     )

    #     trainer.train()
    #     f1 = trainer.evaluate()["eval_f1"]
    #     if f1 > best_f1:
    #         best_f1 = f1

    #     if trainer.state.best_model_checkpoint:
    #         final_best_model_dir = os.path.join(output_dir, config["best_model_path"])
    #         os.makedirs(final_best_model_dir, exist_ok=True)
    #         torch.save(trainer.model.state_dict(), os.path.join(final_best_model_dir, "pytorch_model.bin"))
    #         logger.info(f"✅ 自定义模型权重已保存到 {final_best_model_dir}")

    #         for fname in os.listdir(output_dir):
    #             if fname.startswith("checkpoint") and os.path.isdir(os.path.join(output_dir, fname)):
    #                 shutil.rmtree(os.path.join(output_dir, fname))

    # logger.info(f"[{target}] Best Fold F1: {best_f1}")

    # ✅ 使用统一 label2idx 加载测试集
    test_ds = HostDataset(test_path, tokenizer, max_len, label2idx)

    logger.info(f"\n[🔍 Test on {target}] Start testing each best_model on test.csv")
    fold_metrics = []  # 每折存多指标

    for fold in range(1, 6):
        logger.info(f"\n[Fold {fold}]")

        model_dir = os.path.join(config["save_dir"], target, f"fold{fold}", config["best_model_path"])
        model = load_model(model_dir, num_labels=len(label2idx))
        trainer = Trainer(model=model)

        pred_output = trainer.predict(test_ds)
        logits = pred_output.predictions
        labels = pred_output.label_ids
        preds = np.argmax(logits, axis=1)

        # 仅对测试中出现的类计算指标，防止缺项报错
        existing_labels = sorted(set(preds) | set(labels))
        target_names = [idx2label[i] for i in existing_labels]

        # 基础分类指标
        macro_f1 = f1_score(labels, preds, labels=existing_labels, average="macro")
        mcc = matthews_corrcoef(labels, preds)

        # 概率（用于 AUC）
        probs = torch.softmax(torch.tensor(logits), dim=1).numpy()

        # AUC：若仅单一类别则置 NaN
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

        # 日志
        logger.info(f"Macro F1-score: {macro_f1:.4f}")
        logger.info(f"MCC: {mcc:.4f}")
        logger.info(f"ROC-AUC (macro, OvR): {roc_auc_macro if not np.isnan(roc_auc_macro) else 'NaN'}")
        logger.info(f"PR-AUC  (macro): {pr_auc_macro if not np.isnan(pr_auc_macro) else 'NaN'}")

        cm = confusion_matrix(labels, preds, labels=existing_labels)
        logger.info("Confusion Matrix:\n" + str(cm))
        report = classification_report(labels, preds, labels=existing_labels, target_names=target_names)
        logger.info("Classification Report:\n" + report)

        fold_metrics.append({
            "fold": fold,
            "macro_f1": macro_f1,
            "mcc": mcc,
            "roc_auc_macro": roc_auc_macro,
            "pr_auc_macro": pr_auc_macro
        })

    # === 保存 CSV：每折指标 + mean 行 ===
    df_metrics = pd.DataFrame(fold_metrics)
    mean_row = {
        "fold": "mean",
        "macro_f1": np.nanmean(df_metrics["macro_f1"]),
        "mcc": np.nanmean(df_metrics["mcc"]),
        "roc_auc_macro": np.nanmean(df_metrics["roc_auc_macro"]),
        "pr_auc_macro": np.nanmean(df_metrics["pr_auc_macro"]),
    }
    df_metrics = pd.concat([df_metrics, pd.DataFrame([mean_row])], ignore_index=True)

    result_csv_path = os.path.join(config["save_dir"], target, config["f1_csv"])
    df_metrics.to_csv(result_csv_path, index=False)
    logger.info(f"[Saved] Test metrics by fold (with mean) saved to: {result_csv_path}")

    # 终端总结
    logger.info(
        f"\n[Summary] {target} 5-Fold Test "
        f"F1 Mean: {mean_row['macro_f1']:.4f}, "
        f"MCC Mean: {mean_row['mcc']:.4f}, "
        f"ROC-AUC Mean: {mean_row['roc_auc_macro'] if not np.isnan(mean_row['roc_auc_macro']) else 'NaN'}, "
        f"PR-AUC Mean: {mean_row['pr_auc_macro'] if not np.isnan(mean_row['pr_auc_macro']) else 'NaN'}"
    )

if __name__ == "__main__":
    targets = ["VP4"]
    for target in targets:
        train_for_target(target)
