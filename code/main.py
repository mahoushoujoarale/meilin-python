import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from data_preprocessing import load_data, prepare_features, get_sampling_strategies
from models import get_models
from model_training import evaluate_all_models, find_best_model
from visualization import (
    create_plot_directory, plot_roc_curves, plot_pr_curves,
    plot_shap_values, plot_calibration_curve, plot_decision_curve,
    plot_bootstrap_distributions
)
from utils import (
    setup_device, generate_baseline_table, save_results_to_excel,
    perform_bootstrap_validation, print_bootstrap_results,
    save_external_validation_results
)
from constants import input_files, sheet_names, id_cols

# 设置中文字体和图片样式
plt.rcParams['font.sans-serif'] = ['Hiragino Sans GB']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.figsize'] = [10, 6]
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10

# 设置随机种子
np.random.seed(42)

def main():
    # 设置设备
    device = setup_device()
    
    # 读取数据集
    print("\n读取数据集...")
    df = pd.read_excel("temp/imputed-baseline.xlsx")
    
    # 从文件读取特征列表
    print("\n读取特征列表...")
    with open("temp/selected_features.txt", "r") as f:
        selected_features = [line.strip() for line in f if line.strip()]
    print(f"特征: {selected_features}")
    
    # 生成基线特征表格
    print("\n生成基线特征表格...")
    baseline_df = generate_baseline_table(df, selected_features)
    
    # 创建保存图片的文件夹
    plot_dir = create_plot_directory()
    print(f"\n所有图片将保存到文件夹: {plot_dir}")
    
    # 加载和准备数据
    df = load_data(input_files, sheet_names)
    X_train, X_test, y_train, y_test, scaler = prepare_features(df, selected_features)
    
    # 获取采样策略和模型
    sampling_strategies = get_sampling_strategies()
    models = get_models(X_train.shape[1], y_train)
    
    # 评估所有模型
    results_data, model_scores = evaluate_all_models(
        models, X_train, y_train, X_test, y_test, sampling_strategies, device
    )
    
    # 保存结果到Excel
    save_results_to_excel(results_data)
    
    # 为每个采样策略绘制ROC和PR曲线
    for strategy_name in model_scores.keys():
        plot_roc_curves(model_scores, strategy_name, plot_dir)
        plot_pr_curves(model_scores, strategy_name, plot_dir)
    
    # 找出并评估最佳模型
    best_model, y_pred_proba, model_name = find_best_model(
        model_scores, models, sampling_strategies, X_train, y_train, X_test, y_test, device
    )
    
    # 绘制SHAP值图（仅适用于非神经网络模型）
    if "Neural Network" not in model_name:
        plot_shap_values(best_model, X_test, selected_features, plot_dir)
    
    # 绘制校准曲线和决策曲线
    plot_calibration_curve(y_test, y_pred_proba, plot_dir)
    plot_decision_curve(y_test, y_pred_proba, plot_dir)
    
    # 执行Bootstrap验证
    bootstrap_metrics = perform_bootstrap_validation(
        best_model, X_test, y_test,
        is_neural_net="Neural Network" in model_name,
        device=device
    )
    
    # 打印Bootstrap结果
    print_bootstrap_results(bootstrap_metrics)
    
    # 绘制Bootstrap分布图
    plot_bootstrap_distributions(bootstrap_metrics, plot_dir)
    
    # 保存外部验证结果
    save_external_validation_results(bootstrap_metrics, plot_dir)
    
    print(f"\n所有图片和外部验证结果已保存到文件夹: {plot_dir}")

if __name__ == "__main__":
    main()