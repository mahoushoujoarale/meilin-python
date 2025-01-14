import pandas as pd
from sklearn.feature_selection import RFE, SelectFromModel, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import os
import logging
from constants import baseline_input_files, baseline_sheet_names, id_cols
from catboost import CatBoostClassifier  # 导入 CatBoostClassifier

# 配置日志记录
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def merge_data():
    """
    合并所有归一化后的数据文件
    
    返回:
    pd.DataFrame: 合并后的数据
    """
    merged_df = pd.DataFrame()
    
    for file, sheet in zip(baseline_input_files, baseline_sheet_names):
        input_file = "temp/normalized-" + file
        file_path = os.path.abspath(input_file)
        
        # 检查文件是否存在
        if not os.path.exists(file_path):
            logging.warning(f"文件 {file_path} 不存在，跳过")
            continue
        
        try:
            # 读取Excel文件
            df = pd.read_excel(file_path, sheet_name=sheet)
            merged_df = pd.concat([merged_df, df], ignore_index=True)
        except Exception as e:
            logging.error(f"读取文件 {file_path} 时出错: {e}")
            continue
    
    logging.info("所有数据文件已合并")
    return merged_df

def feature_selection(df, target_column, method="rfe", n_features=7):
    """
    特征筛选
    
    参数:
    df (pd.DataFrame): 合并后的数据
    target_column (str): 目标列名
    method (str): 特征筛选方法，可选 "rfe"（递归特征消除）、"l1"（L1正则化）、"mi"（互信息）、"rf"（随机森林）
    n_features (int): 需要筛选的特征数量
    
    返回:
    list: 筛选出的特征名列表
    """
    X = df.drop(columns=[target_column] + id_cols)
    y = df[target_column]
    
    pos_weight = len(y[y == 0]) / len(y[y == 1])  # 计算类别权重
    
    if method == "rfe":
        model = CatBoostClassifier(iterations=100, random_state=42, verbose=False)  # 使用 CatBoostClassifier
        rfe = RFE(estimator=model, n_features_to_select=n_features)
        rfe.fit(X, y)
        selected_features = X.columns[rfe.support_].tolist()
    elif method == "l1":
        model = LogisticRegression(penalty="l1", solver="liblinear", class_weight="balanced", random_state=42)
        selector = SelectFromModel(model, max_features=n_features)
        selector.fit(X, y)
        selected_features = X.columns[selector.get_support()].tolist()
    elif method == "mi":
        mi_scores = mutual_info_classif(X, y, random_state=42)
        selected_features = X.columns[mi_scores.argsort()[-n_features:]].tolist()
    elif method == "rf":
        model = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
        model.fit(X, y)
        importances = model.feature_importances_
        selected_features = X.columns[importances.argsort()[-n_features:]].tolist()
    else:
        raise ValueError(f"不支持的特征筛选方法: {method}")
    
    logging.info(f"使用 {method} 方法筛选出的 {n_features} 个特征: {selected_features}")
    return selected_features

def evaluate_model(df, target_column, selected_features):
    """
    评估模型性能
    
    参数:
    df (pd.DataFrame): 合并后的数据
    target_column (str): 目标列名
    selected_features (list): 筛选出的特征列表
    """
    X = df[selected_features]
    y = df[target_column]
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # 训练模型
    model = CatBoostClassifier(iterations=100, random_state=42, verbose=False)  # 使用 CatBoostClassifier
    model.fit(X_train, y_train)
    
    # 评估模型
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

def main():
    # 合并数据
    merged_df = merge_data()
    
    if merged_df.empty:
        logging.error("没有数据可处理")
        return
    
    # 设置目标列（根据实际情况修改）
    target_column = "不良反应"
    
    if target_column not in merged_df.columns:
        logging.error(f"目标列 {target_column} 不存在")
        return
    
    # 使用特征筛选方法（可选 "rfe", "l1", "mi"）
    selected_features = feature_selection(merged_df, target_column, method="rfe", n_features=1)
    
    # 评估模型性能
    evaluate_model(merged_df, target_column, selected_features)
    
    # 保存筛选结果
    output_file = os.path.abspath("temp/selected_features.txt")
    with open(output_file, "w") as f:
        for feature in selected_features:
            f.write(feature + "\n")
    logging.info(f"筛选结果已保存到 {output_file}")

if __name__ == "__main__":
    main()