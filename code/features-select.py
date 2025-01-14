import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.ensemble import GradientBoostingClassifier
import os
# from constants import baseline_input_files, baseline_sheet_names, id_cols
from constants import comparison_input_files, comparison_sheet_names, id_cols

def merge_data():
    """
    合并所有归一化后的数据文件
    
    返回:
    pd.DataFrame: 合并后的数据
    """
    merged_df = pd.DataFrame()
    
    # for file, sheet in zip(baseline_input_files, baseline_sheet_names):
    for file, sheet in zip(comparison_input_files, comparison_sheet_names):
        input_file = "temp/normalized-" + file
        file_path = os.path.abspath(input_file)
        
        # 检查文件是否存在
        if not os.path.exists(file_path):
            print(f"文件 {file_path} 不存在，请检查路径")
            continue
        
        # 读取Excel文件
        df = pd.read_excel(file_path, sheet_name=sheet)
        merged_df = pd.concat([merged_df, df], ignore_index=True)
    
    print("所有数据文件已合并")
    return merged_df

def feature_selection_with_rfe(df, target_column, n_features=7):
    """
    使用递归特征消除（RFE）进行特征筛选
    
    参数:
    df (pd.DataFrame): 合并后的数据
    target_column (str): 目标列名
    n_features (int): 需要筛选的特征数量
    
    返回:
    list: 筛选出的特征名列表
    """
    # 分离特征和目标变量
    X = df.drop(columns=[target_column] + id_cols)
    y = df[target_column]
    
    # 选择模型（这里使用梯度提升树）
    model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    
    # 使用 RFE 进行特征筛选
    rfe = RFE(estimator=model, n_features_to_select=n_features)
    rfe.fit(X, y)
    
    # 获取筛选出的特征
    selected_features = X.columns[rfe.support_].tolist()
    
    return selected_features

def main():
    # 合并数据
    merged_df = merge_data()
    
    if merged_df.empty:
        print("没有数据可处理")
        return
    
    # 设置目标列（根据实际情况修改）
    target_column = "不良反应"
    
    if target_column not in merged_df.columns:
        print(f"目标列 {target_column} 不存在")
        return
    
    # 使用 RFE 进行特征筛选
    selected_features = feature_selection_with_rfe(merged_df, target_column, n_features=7)
    
    # 打印筛选出的特征
    print("筛选出的 7 个最重要特征：")
    print(selected_features)
    
    # 保存筛选结果
    output_file = os.path.abspath("temp/selected_features_rfe.txt")
    with open(output_file, "w") as f:
        for feature in selected_features:
            f.write(feature + "\n")
    print(f"筛选结果已保存到 {output_file}")

if __name__ == "__main__":
    main()