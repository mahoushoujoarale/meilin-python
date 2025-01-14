import pandas as pd
from sklearn.preprocessing import RobustScaler  # 导入 RobustScaler
import os
from constants import binary_cols, id_cols, input_files, sheet_names

def normalize_data(file_name, sheet_name):
    input_file = "temp/imputed-" + file_name
    output_file = "temp/normalized-" + file_name

    """
    对数据进行归一化和二分类调整
    
    参数:
    file_name (str): 插补后的Excel文件名
    sheet_name (str): 工作表名
    """
    # 将路径转换为绝对路径
    file_path = os.path.abspath(input_file)
    output_path = os.path.abspath(output_file)
    
    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"文件 {file_path} 不存在，请检查路径")
        return
    
    # 读取Excel文件
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    
    # 对二分类列调整为0和1
    if binary_cols:
        for col in binary_cols:
            if col in df.columns:
                # 将二分类列的值映射为0和1
                df[col] = df[col].astype(int).replace({1: 1, 2: 0})
                print(f"二分类列 {col} 已调整为0和1")
            else:
                print(f"列 {col} 不存在，跳过")
    
    # 对连续型列进行归一化
    numeric_cols = df.select_dtypes(include=['number']).columns.difference(id_cols).difference(binary_cols)
    if len(numeric_cols) > 0:
        scaler = RobustScaler()  # 使用 RobustScaler 替换 MinMaxScaler
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        print("连续型列已使用 RobustScaler 归一化")
    else:
        print("没有连续型列，无需归一化")
    
    # 保存归一化后的数据
    df.to_excel(output_path, index=False, sheet_name=sheet_name)
    print(f"归一化完成，结果保存到 {output_path}")

# 对每个文件进行处理
for file, sheet in zip(input_files, sheet_names):
    normalize_data(file, sheet)