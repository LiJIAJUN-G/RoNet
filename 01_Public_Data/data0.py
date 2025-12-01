import pandas as pd

def filter_csv_by_protein_and_length(input_path, output_path, keywords, len_min=0, len_max=float('inf'), dropna_host=False):
    """
    过滤CSV中Protein列含有指定关键词，且Length在给定范围内的行，并可选择删除Host为NaN的行，保存为新CSV。

    参数：
    - input_path: 输入CSV文件路径
    - output_path: 输出CSV文件路径
    - keywords: 包含的关键词列表，例如 ["VP4", "VP7"]
    - len_min: Length的最小值（包含）
    - len_max: Length的最大值（包含）
    - dropna_host: 是否删除 Host 为 NaN 的行
    """
    # 读取CSV文件
    df = pd.read_csv(input_path)

    # 过滤 Protein 列中包含任意关键词的行
    pattern = '|'.join(keywords)
    df_filtered = df[df['Protein'].str.contains(pattern, na=False)]

    # 过滤 Length 在指定范围内的行
    df_filtered = df_filtered[(df_filtered['Length'] >= len_min) & (df_filtered['Length'] <= len_max)]

    # 可选：删除 Host 列为 NaN 的行
    if dropna_host and 'Host' in df_filtered.columns:
        df_filtered = df_filtered[df_filtered['Host'].notna()]

    print(f"筛选后行数: {len(df_filtered)}")
    df_filtered.to_csv(output_path, index=False)
    print(f"已保存至: {output_path}")

# 调用示例
filter_csv_by_protein_and_length(
    input_path="raw_data/sequences_2018.csv",
    output_path="pro_data/pretraining_2018_VP.csv",
    keywords=["VP"],
    len_min=300,
    len_max=1150,
    dropna_host=False  # 保留Host为NaN的行
)

filter_csv_by_protein_and_length(
    input_path="raw_data/sequences_2018.csv",
    output_path="pro_data/training_2018_VP4.csv",
    keywords=["VP4"],
    len_min=700,
    len_max=850,
    dropna_host=True
)

filter_csv_by_protein_and_length(
    input_path="raw_data/sequences_2018.csv",
    output_path="pro_data/training_2018_VP7.csv",
    keywords=["VP7"],
    len_min=300,
    len_max=360,
    dropna_host=True
)

filter_csv_by_protein_and_length(
    input_path="raw_data/sequences_2018-2025.csv",
    output_path="pro_data/testing_2018-2025_VP4.csv",
    keywords=["VP4"],
    len_min=700,
    len_max=850,
    dropna_host=True
)

filter_csv_by_protein_and_length(
    input_path="raw_data/sequences_2018-2025.csv",
    output_path="pro_data/testing_2018-2025_VP7.csv",
    keywords=["VP7"],
    len_min=300,
    len_max=360,
    dropna_host=True
)
