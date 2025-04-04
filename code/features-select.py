import pandas as pd
from sklearn.feature_selection import RFE, SelectFromModel, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import os
import logging
from constants import input_files, sheet_names, id_cols
from catboost import CatBoostClassifier  # 导入 CatBoostClassifier
from sklearn.linear_model import Lasso
from scipy import stats as scipy_stats  # 重命名为 scipy_stats
from scipy.stats import chi2_contingency
import numpy as np
from sklearn.preprocessing import StandardScaler

# 配置日志记录
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def merge_data():
    """
    合并所有归一化后的数据文件
    
    返回:
    pd.DataFrame: 合并后的数据
    """
    merged_df = pd.DataFrame()
    
    for file, sheet in zip(input_files, sheet_names):
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

def analyze_feature_types(df, target_column):
    """
    分析特征类型，将特征分为二分类变量和连续变量
    
    参数:
    df (pd.DataFrame): 数据框
    target_column (str): 目标列名
    
    返回:
    tuple: (二分类变量列表, 连续变量列表)
    """
    binary_features = []
    continuous_features = []
    
    X = df.drop(columns=[target_column] + id_cols)
    
    for column in X.columns:
        unique_values = df[column].nunique()
        if unique_values == 2 or (df[column].dtype in ['object', 'category'] and unique_values <= 2):
            binary_features.append(column)
        elif df[column].dtype in ['int64', 'float64'] and unique_values > 2:
            continuous_features.append(column)
    
    return binary_features, continuous_features

def calculate_statistics(df, target_column, binary_features, continuous_features):
    """
    计算特征的统计量
    
    参数:
    df (pd.DataFrame): 数据框
    target_column (str): 目标列名
    binary_features (list): 二分类变量列表
    continuous_features (list): 连续变量列表
    
    返回:
    dict: 特征统计信息
    """
    feature_stats = {}  # 改名为 feature_stats 避免与模块名冲突
    y = df[target_column]
    
    # 处理二分类变量
    for feature in binary_features:
        contingency_table = pd.crosstab(df[feature], y)
        chi2, p_value, _, _ = chi2_contingency(contingency_table)
        odds_ratio = (contingency_table.iloc[1,1] * contingency_table.iloc[0,0]) / \
                    (contingency_table.iloc[0,1] * contingency_table.iloc[1,0])
        feature_stats[feature] = {
            'type': 'binary',
            'p_value': p_value,
            'chi2': chi2,
            'odds_ratio': odds_ratio
        }
    
    # 处理连续变量
    for feature in continuous_features:
        # Mann-Whitney U 检验
        positive_group = df[df[target_column] == 1][feature].dropna()
        negative_group = df[df[target_column] == 0][feature].dropna()
        statistic, p_value = scipy_stats.mannwhitneyu(positive_group, negative_group, alternative='two-sided')
        
        # 计算效应量 (Cohen's d)
        cohens_d = (positive_group.mean() - negative_group.mean()) / \
                  np.sqrt((positive_group.var() + negative_group.var()) / 2)
        
        feature_stats[feature] = {
            'type': 'continuous',
            'p_value': p_value,
            'effect_size': abs(cohens_d),
            'mean_diff': positive_group.mean() - negative_group.mean()
        }
    
    return feature_stats

def select_significant_features(stats, alpha=0.05):
    """
    根据统计显著性和效应量选择特征
    
    参数:
    stats (dict): 特征统计信息
    alpha (float): 显著性水平
    
    返回:
    list: 显著特征列表
    """
    significant_features = []
    
    # 分别存储二分类和连续变量的特征及其重要性分数
    binary_scores = []
    continuous_scores = []
    
    for feature, info in stats.items():
        if info['p_value'] < alpha:
            if info['type'] == 'binary':
                binary_scores.append((feature, abs(np.log(info['odds_ratio']))))
            else:
                continuous_scores.append((feature, info['effect_size']))
    
    # 对每种类型的特征分别排序
    binary_scores.sort(key=lambda x: x[1], reverse=True)
    continuous_scores.sort(key=lambda x: x[1], reverse=True)
    
    # 从每种类型中选择最重要的特征
    significant_features.extend([f[0] for f in binary_scores])
    significant_features.extend([f[0] for f in continuous_scores])
    
    return significant_features

def feature_selection(df, target_column, method="rfe", n_features=7):
    """
    特征筛选
    
    参数:
    df (pd.DataFrame): 合并后的数据
    target_column (str): 目标列名
    method (str): 特征筛选方法
    n_features (int): 需要筛选的特征数量
    
    返回:
    list: 筛选出的特征名列表
    """
    # 分析特征类型
    binary_features, continuous_features = analyze_feature_types(df, target_column)
    logging.info(f"二分类变量: {len(binary_features)}个")
    logging.info(f"连续变量: {len(continuous_features)}个")
    
    # 计算统计量
    stats = calculate_statistics(df, target_column, binary_features, continuous_features)
    
    # 选择显著特征
    significant_features = select_significant_features(stats)
    logging.info(f"显著特征数量: {len(significant_features)}")
    
    # 输出每个显著特征的统计信息
    for feature in significant_features:
        info = stats[feature]
        if info['type'] == 'binary':
            logging.info(f"特征 {feature} (二分类): p值={info['p_value']:.4f}, OR={info['odds_ratio']:.2f}")
        else:
            logging.info(f"特征 {feature} (连续): p值={info['p_value']:.4f}, 效应量={info['effect_size']:.2f}")
    
    if not significant_features:
        logging.warning("没有找到显著特征")
        return []
    
    # 对显著特征进行进一步筛选
    X = df[significant_features]
    y = df[target_column]
    
    if method == "rfe":
        model = CatBoostClassifier(iterations=100, random_state=42, verbose=False)
        rfe = RFE(estimator=model, n_features_to_select=min(n_features, len(significant_features)))
        rfe.fit(X, y)
        selected_features = X.columns[rfe.support_].tolist()
    elif method == "l1":
        model = LogisticRegression(penalty="l1", solver="liblinear", class_weight="balanced", random_state=42)
        selector = SelectFromModel(model, max_features=min(n_features, len(significant_features)))
        selector.fit(X, y)
        selected_features = X.columns[selector.get_support()].tolist()
    elif method == "mi":
        mi_scores = mutual_info_classif(X, y, random_state=42)
        selected_features = X.columns[mi_scores.argsort()[-min(n_features, len(significant_features)):]].tolist()
    elif method == "rf":
        model = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
        model.fit(X, y)
        importances = model.feature_importances_
        selected_features = X.columns[importances.argsort()[-min(n_features, len(significant_features)):]].tolist()
    elif method == "lasso":
        # 对连续变量进行标准化
        scaler = StandardScaler()
        X_scaled = X.copy()
        for feature in continuous_features:
            if feature in X.columns:
                X_scaled[feature] = scaler.fit_transform(X[[feature]])
        
        lasso = Lasso(alpha=0.01, random_state=42)
        lasso.fit(X_scaled, y)
        selected_features = X.columns[lasso.coef_ != 0].tolist()
        if len(selected_features) > n_features:
            selected_features = X.columns[abs(lasso.coef_).argsort()[-n_features:]].tolist()
    else:
        raise ValueError(f"不支持的特征筛选方法: {method}")
    
    # 输出最终选择的特征及其统计信息
    logging.info("\n最终选择的特征:")
    for feature in selected_features:
        info = stats[feature]
        if info['type'] == 'binary':
            logging.info(f"特征 {feature} (二分类):")
            logging.info(f"  - p值: {info['p_value']:.4f}")
            logging.info(f"  - 优势比(OR): {info['odds_ratio']:.2f}")
        else:
            logging.info(f"特征 {feature} (连续):")
            logging.info(f"  - p值: {info['p_value']:.4f}")
            logging.info(f"  - 效应量: {info['effect_size']:.2f}")
            logging.info(f"  - 均值差: {info['mean_diff']:.2f}")
    
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
    model = CatBoostClassifier(iterations=100, random_state=42, verbose=False)
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
    selected_features = feature_selection(merged_df, target_column, method="rfe", n_features=7)
    
    if selected_features:
        # 评估模型性能
        evaluate_model(merged_df, target_column, selected_features)
        
        # 保存筛选结果
        output_file = os.path.abspath("temp/selected_features.txt")
        with open(output_file, "w") as f:
            for feature in selected_features:
                f.write(feature + "\n")
        logging.info(f"筛选结果已保存到 {output_file}")
    else:
        logging.warning("没有找到显著特征，无法进行模型评估")

if __name__ == "__main__":
    main()