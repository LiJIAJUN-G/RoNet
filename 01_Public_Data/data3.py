import pandas as pd
import os

def process_file(file_path, output_path, host_map):
    """处理CSV文件：重新分类 Host 与 Protein 并保存过滤结果"""
    df = pd.read_csv(file_path)
    df = df[['Accession', 'Host', 'Collection_Date']]

    # 映射 Host，过滤非目标宿主
    df['Host'] = df['Host'].map(host_map)
    df = df[df['Host'].notna()]


    # 保存结果
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    # 输出统计
    print(f"\nProcessed: {file_path}")
    print(df['Host'].value_counts())
    print(f"Filtered rows count: {df.shape[0]}")

# 统一宿主映射（5类）
# 标准宿主映射（使用系统发育分类科级名称，常用于学术论文）
host_map = {
    "Homo sapiens": "Hominidae",              # 人科
    "Sus scrofa": "Suidae",                   # 猪科
    "Sus scrofa domesticus": "Suidae",        # 家猪
    "Bos taurus": "Bovidae",                  # 牛科
    "Bos grunniens": "Bovidae",               # 牦牛
    "Bovidae": "Bovidae",                     # 牛科（已为科级名）
    "Equus caballus": "Equidae",              # 马科
    "Gallus gallus": "Phasianidae"            # 雉科（家鸡）
}

def process_file2(file_path, output_path, host_map):
    """处理CSV文件：重新分类 Host 与 Protein 并保存过滤结果"""
    df = pd.read_csv(file_path)
    df = df[['Isolate', 'Accession_VP4', 'Accession_VP7', 'Host', 'Collection_Date_VP4', 'Collection_Date_VP7']]

    # 映射 Host，过滤非目标宿主
    df['Host'] = df['Host'].map(host_map)
    df = df[df['Host'].notna()]


    # 保存结果
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    # 输出统计
    print(f"\nProcessed: {file_path}")
    print(df['Host'].value_counts())
    print(f"Filtered rows count: {df.shape[0]}")


# 执行处理
process_file2(
    file_path="pro_data/testing_2018-2025_VP4_VP7.csv",
    output_path="fin_data/fin_testing_2018-2025_VP4_VP7.csv",
    host_map=host_map
)

process_file2(
    file_path="pro_data/training_2018_VP4_VP7.csv",
    output_path="fin_data/fin_training_2018_VP4_VP7.csv",
    host_map=host_map
)

process_file(
    file_path="pro_data/testing_2018-2025_VP4.csv",
    output_path="fin_data/fin_testing_2018-2025_VP4.csv",
    host_map=host_map
)

process_file(
    file_path="pro_data/testing_2018-2025_VP7.csv",
    output_path="fin_data/fin_testing_2018-2025_VP7.csv",
    host_map=host_map
)

process_file(
    file_path="pro_data/training_2018_VP4.csv",
    output_path="fin_data/fin_training_2018_VP4.csv",
    host_map=host_map
)

process_file(
    file_path="pro_data/training_2018_VP7.csv",
    output_path="fin_data/fin_training_2018_VP7.csv",
    host_map=host_map
)
