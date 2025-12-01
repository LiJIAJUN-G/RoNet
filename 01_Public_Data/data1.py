import pandas as pd

def merge_vp4_vp7_by_isolate(vp4_path, vp7_path, output_path):
    """
    合并 VP4 和 VP7 CSV 文件（按 Isolate 匹配），并判断 Host 和 Collection_Date 是否相等，
    若相等则合并为一列，否则保留两列。

    参数：
    - vp4_path: VP4 的 CSV 文件路径
    - vp7_path: VP7 的 CSV 文件路径
    - output_path: 合并后 CSV 的保存路径
    """
    # 读取 CSV 文件，仅保留指定列
    vp4 = pd.read_csv(vp4_path, usecols=["Accession", "Isolate", "Host", "Collection_Date"])
    vp7 = pd.read_csv(vp7_path, usecols=["Accession", "Isolate", "Host", "Collection_Date"])

    # 重命名列，避免合并后冲突
    vp4 = vp4.rename(columns={
        "Accession": "Accession_VP4",
        "Host": "Host_VP4",
        "Collection_Date": "Collection_Date_VP4"
    })
    vp7 = vp7.rename(columns={
        "Accession": "Accession_VP7",
        "Host": "Host_VP7",
        "Collection_Date": "Collection_Date_VP7"
    })

    # 删除缺失 Isolate 的行
    vp4 = vp4.dropna(subset=["Isolate"])
    vp7 = vp7.dropna(subset=["Isolate"])

    # 多对多合并
    merged = pd.merge(vp4, vp7, on="Isolate", how="inner")

    # 合并 Host 列（如相同）
    if "Host_VP4" in merged.columns and "Host_VP7" in merged.columns:
        if (merged["Host_VP4"] == merged["Host_VP7"]).all():
            print("Host 列一致，保留一列。")
            merged = merged.drop(columns=["Host_VP7"])
            merged = merged.rename(columns={"Host_VP4": "Host"})
        else:
            print("Host 列不一致，保留两列。")

    # 合并 Collection_Date 列（如相同）
    if "Collection_Date_VP4" in merged.columns and "Collection_Date_VP7" in merged.columns:
        if (merged["Collection_Date_VP4"] == merged["Collection_Date_VP7"]).all():
            print("Collection_Date 列一致，保留一列。")
            merged = merged.drop(columns=["Collection_Date_VP7"])
            merged = merged.rename(columns={"Collection_Date_VP4": "Collection_Date"})
        else:
            print("Collection_Date 列不一致，保留两列。")

    # 输出信息与保存文件
    print(f"共有 Isolate 数量: {merged['Isolate'].nunique()}")
    print(f"组合总数: {len(merged)}")
    print(merged.head())
    merged.to_csv(output_path, index=False)
    print(f"已保存至: {output_path}")

merge_vp4_vp7_by_isolate(
    vp4_path="pro_data/testing_2018-2025_VP4.csv",
    vp7_path="pro_data/testing_2018-2025_VP7.csv",
    output_path="pro_data/testing_2018-2025_VP4_VP7.csv"
)

merge_vp4_vp7_by_isolate(
    vp4_path="pro_data/training_2018_VP4.csv",
    vp7_path="pro_data/training_2018_VP7.csv",
    output_path="pro_data/training_2018_VP4_VP7.csv"
)

