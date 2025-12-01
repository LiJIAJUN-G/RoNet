import pandas as pd

# 文件路径与对应的输出文件名映射
files = {
    "testing_2018-2025_VP4.csv": "host_counts_testing_VP4.csv",
    "testing_2018-2025_VP7.csv": "host_counts_testing_VP7.csv",
    "training_2018_VP4.csv": "host_counts_training_VP4.csv",
    "training_2018_VP7.csv": "host_counts_training_VP7.csv",
    "testing_2018-2025_VP4_VP7.csv": "host_counts_testing_VP4_VP7.csv",
    "training_2018_VP4_VP7.csv": "host_counts_training_VP4_VP7.csv"
}

# 遍历每个文件进行统计并保存
for input_file, output_file in files.items():
    df = pd.read_csv(f"pro_data/{input_file}")
    df['Host'].value_counts().to_csv(f"pro_data/{output_file}", header=True)
    print(f"{input_file} 的 Host 列频数已保存到 {output_file}")
