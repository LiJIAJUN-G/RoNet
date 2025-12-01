# pretrain_mlm.py
import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import (
    EsmTokenizer,
    EsmForMaskedLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

# === 参数配置 ===
PRETRAIN_CSV = "../00_Data/Pretraining_VP.csv"  
MODEL_NAME = "../esm_model/esm2_t6_8M_UR50D"
SAVE_DIR = "esm_pretrain_output"
BATCH_SIZE = 8
EPOCHS = 10
LR = 5e-5
MAX_LEN = 1150
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# === 自定义 Dataset ===
class SeqDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_len):
        self.data = pd.read_csv(csv_file)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.sequences = self.data["Seq"].tolist()

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.sequences[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        return {k: v.squeeze(0) for k, v in encoding.items()}


def pretrain():
    tokenizer = EsmTokenizer.from_pretrained(MODEL_NAME)
    model = EsmForMaskedLM.from_pretrained(MODEL_NAME)

    dataset = SeqDataset(PRETRAIN_CSV, tokenizer, MAX_LEN)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

    training_args = TrainingArguments(
        output_dir=SAVE_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        eval_strategy="no",
        save_strategy="epoch",
        num_train_epochs=EPOCHS,
        learning_rate=LR,
        logging_dir=os.path.join(SAVE_DIR, "logs"),
        logging_steps=10,
        save_total_limit=2,
        remove_unused_columns=False,
        report_to="tensorboard",
        fp16=True if DEVICE == "cuda" else False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model(os.path.join(SAVE_DIR, "final_model"))

if __name__ == "__main__":
    pretrain()
