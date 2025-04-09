import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
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

def process_model(args):
    """处理单个模型的函数，用于并行执行"""
    model_name, scores, strategy_name, model, X_train, y_train, X_test, y_test, selected_features, device, plot_dir, sampling_strategies = args
    
    # 创建模型特定的子目录
    model_plot_dir = os.path.join(plot_dir, f"{strategy_name}_{model_name}")
    os.makedirs(model_plot_dir, exist_ok=True)
    
    # 绘制ROC和PR曲线
    plot_roc_curves({strategy_name: {model_name: scores}}, strategy_name, model_plot_dir)
    plot_pr_curves({strategy_name: {model_name: scores}}, strategy_name, model_plot_dir)
    
    # 如果是非神经网络模型，绘制SHAP值图
    if "Neural Network" not in model_name:
        # 重新训练模型以获取SHAP值
        sampler = sampling_strategies[strategy_name]
        X_train_res, y_train_res = sampler.fit_resample(X_train, y_train)
        model.fit(X_train_res, y_train_res)
        plot_shap_values(model, X_test, selected_features, model_plot_dir)
    
    # 绘制校准曲线和决策曲线
    plot_calibration_curve(y_test, scores['y_pred_proba'], model_plot_dir)
    plot_decision_curve(y_test, scores['y_pred_proba'], model_plot_dir)
    
    # 执行Bootstrap验证
    bootstrap_metrics = perform_bootstrap_validation(
        model, X_test, y_test,
        is_neural_net="Neural Network" in model_name,
        device=device
    )
    
    # 绘制Bootstrap分布图
    plot_bootstrap_distributions(bootstrap_metrics, model_plot_dir)
    
    # 保存外部验证结果到Excel
    external_validation_file = os.path.join(model_plot_dir, f'external_validation_{model_name}.xlsx')
    save_external_validation_results(bootstrap_metrics, external_validation_file)
    
    # 保存模型评估结果到Excel
    model_results = {
        '采样策略': strategy_name,
        '模型': model_name,
        'ROC-AUC': scores['auc'],
        'ROC-AUC 95%CI': f"{scores['auc_ci'][0]:.4f}-{scores['auc_ci'][1]:.4f}",
        'PR-AUC': scores['average_precision'],
        'Accuracy': scores['accuracy'],
        'Sensitivity': scores['recall'],
        'Specificity': scores['specificity'],
        'Precision': scores['precision'],
        'F1-score': scores['f1'],
        'Brier Score': scores['brier_score']
    }
    model_results_df = pd.DataFrame([model_results])
    model_results_file = os.path.join(model_plot_dir, f'model_evaluation_{model_name}.xlsx')
    model_results_df.to_excel(model_results_file, index=False)
    
    return model_name, strategy_name, bootstrap_metrics

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
    
    # 准备并行处理的任务
    tasks = []
    neural_network_tasks = []
    
    for strategy_name, strategy_scores in model_scores.items():
        for model_name, scores in strategy_scores.items():
            task = (
                model_name, scores, strategy_name, models[model_name],
                X_train, y_train, X_test, y_test, selected_features, device,
                plot_dir, sampling_strategies
            )
            if "Neural Network" in model_name:
                neural_network_tasks.append(task)
            else:
                tasks.append(task)
    
    # 首先处理神经网络模型（单进程）
    print("\n处理神经网络模型...")
    for task in neural_network_tasks:
        model_name, strategy_name, bootstrap_metrics = process_model(task)
        print(f"\n{'-'*30}")
        print(f"完成 {strategy_name} 采样策略下的 {model_name} 模型评估")
        print(f"{'-'*30}")
        print_bootstrap_results(bootstrap_metrics)
    
    # 然后并行处理其他模型
    print("\n开始并行处理其他模型...")
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = {executor.submit(process_model, task): task for task in tasks}
        
        for future in as_completed(futures):
            model_name, strategy_name, bootstrap_metrics = future.result()
            print(f"\n{'-'*30}")
            print(f"完成 {strategy_name} 采样策略下的 {model_name} 模型评估")
            print(f"{'-'*30}")
            print_bootstrap_results(bootstrap_metrics)
    
    # 找出并评估最佳模型
    best_model, y_pred_proba, model_name = find_best_model(
        model_scores, models, sampling_strategies, X_train, y_train, X_test, y_test, device
    )
    
    print(f"\n所有图片和结果已保存到文件夹: {plot_dir}")

if __name__ == "__main__":
    main()