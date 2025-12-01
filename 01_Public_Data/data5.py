import pandas as pd
from Bio import SeqIO
import os

# 读取FASTA并映射 accession -> sequence
def load_fasta_sequences(fasta_path):
    acc_to_seq = {}
    for record in SeqIO.parse(fasta_path, "fasta"):
        acc = record.id.split('.')[0]  # 去掉版本号
        acc_to_seq[acc] = str(record.seq)
    return acc_to_seq

# 添加序列列
def add_sequence_column(df, acc_to_seq):
    df["Seq"] = df["Accession"].apply(lambda x: acc_to_seq.get(str(x).split('.')[0], None))
    return df

# 文件路径
csv_path = "pro_data/pretraining_2018_VP.csv"
fasta_path = "raw_data/sequences_2018.fasta"
output_path = "fin_data_seq/pretraining_2018_VP_with_seq.csv"

# 处理流程
df = pd.read_csv(csv_path, usecols=["Accession", "Host", "Collection_Date"])
acc_to_seq = load_fasta_sequences(fasta_path)
df = add_sequence_column(df, acc_to_seq)

# 删除任何包含 NaN 的行
df = df.dropna()

# 保存结果
df.to_csv(output_path, index=False)
print(f"✔ 已保存：{output_path}")
