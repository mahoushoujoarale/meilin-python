import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTETomek

def load_data(input_files, sheet_names):
    """加载数据集"""
    df = pd.DataFrame()
    for file, sheet in zip(input_files, sheet_names):
        input_file = "temp/normalized-" + file
        current_df = pd.read_excel(input_file, sheet_name=sheet)
        print(f"文件 {input_file} 读取成功")
        df = pd.concat([df, current_df], ignore_index=True)
    return df

def prepare_features(df, selected_features):
    """准备特征数据"""
    X = df[selected_features]
    y = df['不良反应']
    
    # 对分类变量进行编码
    X = pd.get_dummies(X)
    
    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # 特征缩放
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test, scaler

def get_sampling_strategies():
    """获取采样策略"""
    return {
        "SMOTE": SMOTE(random_state=42),
        "ADASYN": ADASYN(random_state=42),
        "SMOTE_Tomek": SMOTETomek(random_state=42)
    }

def apply_sampling(X_train, y_train, sampler):
    """应用采样策略"""
    return sampler.fit_resample(X_train, y_train) 