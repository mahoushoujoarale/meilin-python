import pandas as pd
import os

def calculate_missing_values(file_name, output_file):
    """
    计算Excel文件中每一列的缺失值比例，并将结果保存到新文件
    
    参数:
    file_name (str): 原始Excel文件名（支持相对路径或绝对路径）
    output_file (str): 输出文件名（支持相对路径或绝对路径）
    """
    # 将路径转换为绝对路径
    file_path = os.path.abspath(file_name)
    output_path = os.path.abspath(output_file)
    
    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"文件 {file_path} 不存在，请检查路径")
        return
    
    # 读取Excel文件
    df = pd.read_excel(file_path, sheet_name='基线阴性插补前')
    
    # 计算每一列的缺失值比例
    missing_ratio = df.isnull().mean()  # 计算缺失值比例
    missing_ratio = missing_ratio * 100  # 转换为百分比
    
    # 将结果保存到DataFrame
    result_df = pd.DataFrame({
        'Column': missing_ratio.index,
        'Missing Ratio (%)': missing_ratio.values
    })
    
    # 保存结果到新文件
    result_df.to_excel(output_path, index=False)
    print(f"缺失值比例已计算，结果保存到 {output_path}")

# 示例：计算缺失值比例
calculate_missing_values(
    file_name='input/negative-baseline.xlsx',  # 支持相对路径或绝对路径
    output_file='temp/missing_ratio.xlsx'  # 支持相对路径或绝对路径
)