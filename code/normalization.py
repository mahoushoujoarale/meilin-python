import pandas as pd
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from scipy.stats import boxcox
import os
import numpy as np
from constants import binary_cols, id_cols, input_files, sheet_names

def tanh_normalize(data):
    """
    使用 Tanh 归一化方法对数据进行归一化
    """
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    return np.tanh((data - mean) / std)

def boxcox_normalize(data):
    """
    使用 Box-Cox 归一化方法对数据进行归一化
    """
    # Box-Cox 要求数据必须为正数，因此先平移数据
    data_shifted = data - np.min(data, axis=0) + 1e-5  # 避免零值
    normalized_data = np.zeros_like(data)
    for i in range(data.shape[1]):  # 对每一列单独处理
        normalized_data[:, i], _ = boxcox(data_shifted[:, i])
    return normalized_data

def l1_normalize(data):
    """
    使用 L1 归一化方法对数据进行归一化
    """
    l1_norm = np.sum(np.abs(data), axis=0)  # 计算每一列的L1范数
    return data / l1_norm  # 将每一列的元素除以该列的L1范数

def normalize_data(file_name, sheet_name, normalization_method='robust'):
    """
    对数据进行归一化和二分类调整

    参数:
    file_name (str): 插补后的Excel文件名
    sheet_name (str): 工作表名
    normalization_method (str): 归一化方法，可选 'robust', 'minmax', 'tanh', 'boxcox', 'l1'
    """
    input_file = "temp/imputed-" + file_name
    output_file = "temp/normalized-" + file_name

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
    # if binary_cols:
    #     for col in binary_cols:
    #         if col in df.columns:
    #             # 将二分类列的值映射为0和1
    #             df[col] = df[col].astype(int).replace({1: 1, 2: 0})
    #             print(f"二分类列 {col} 已调整为0和1")
    #         else:
    #             print(f"列 {col} 不存在，跳过")

    # 对连续型列进行归一化
    numeric_cols = df.select_dtypes(include=['number']).columns.difference(id_cols).difference(binary_cols)
    if len(numeric_cols) > 0:
        if normalization_method == 'robust':
            scaler = RobustScaler()
            df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
            print("连续型列已使用 RobustScaler 归一化")
        elif normalization_method == 'minmax':
            scaler = MinMaxScaler()
            df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
            print("连续型列已使用 MinMaxScaler 归一化")
        elif normalization_method == 'tanh':
            df[numeric_cols] = tanh_normalize(df[numeric_cols])
            print("连续型列已使用 Tanh 归一化")
        elif normalization_method == 'boxcox':
            df[numeric_cols] = boxcox_normalize(df[numeric_cols].values)
            print("连续型列已使用 Box-Cox 归一化")
        elif normalization_method == 'l1':
            df[numeric_cols] = l1_normalize(df[numeric_cols].values)
            print("连续型列已使用 L1 归一化")
        else:
            raise ValueError(f"不支持的归一化方法: {normalization_method}")
    else:
        print("没有连续型列，无需归一化")

    # 保存归一化后的数据
    df.to_excel(output_path, index=False, sheet_name=sheet_name)
    print(f"归一化完成，结果保存到 {output_path}")

# 对每个文件进行处理
for file, sheet in zip(input_files, sheet_names):
    normalize_data(file, sheet, normalization_method='tanh')  # 可以通过参数控制归一化方法