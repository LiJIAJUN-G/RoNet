import pandas as pd
from Bio import SeqIO
import os

# 读取FASTA并映射 accession -> sequence
def load_fasta_sequences(fasta_path):
    acc_to_seq = {}
    for record in SeqIO.parse(fasta_path, "fasta"):
        acc = record.id.split('.')[0]
        acc_to_seq[acc] = str(record.seq)
    return acc_to_seq

# 添加序列列
def add_sequence_column(df, col_name, new_col_name, acc_to_seq):
    df[new_col_name] = df[col_name].apply(lambda x: acc_to_seq.get(str(x).split('.')[0], None))
    return df

# 文件配置
file_configs = {
    "fin_testing_2018-2025_VP4_VP7.csv": "raw_data/sequences_2018-2025.fasta",
    "fin_testing_2018-2025_VP4.csv": "raw_data/sequences_2018-2025.fasta",
    "fin_testing_2018-2025_VP7.csv": "raw_data/sequences_2018-2025.fasta",
    "fin_training_2018_VP4_VP7.csv": "raw_data/sequences_2018.fasta",
    "fin_training_2018_VP4.csv": "raw_data/sequences_2018.fasta",
    "fin_training_2018_VP7.csv": "raw_data/sequences_2018.fasta"
}

input_folder = "fin_data"
output_folder = "fin_data_seq"
os.makedirs(output_folder, exist_ok=True)

# 批量处理
for file_name, fasta_path in file_configs.items():
    input_path = os.path.join(input_folder, file_name)
    output_path = os.path.join(output_folder, file_name)

    df = pd.read_csv(input_path)
    acc_to_seq = load_fasta_sequences(fasta_path)

    if "Accession_VP4" in df.columns and "Accession_VP7" in df.columns:
        df = add_sequence_column(df, "Accession_VP4", "VP4_Seq", acc_to_seq)
        df = add_sequence_column(df, "Accession_VP7", "VP7_Seq", acc_to_seq)
    elif "Accession" in df.columns:
        df = add_sequence_column(df, "Accession", "Seq", acc_to_seq)

    # 删除任何包含 NaN 的行
    df = df.dropna()

    df.to_csv(output_path, index=False)
    print(f"✔ 已保存（无NaN）：{output_path}")
