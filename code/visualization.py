import matplotlib.pyplot as plt
import numpy as np
import shap
import os
from datetime import datetime
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.calibration import calibration_curve
from scipy import stats
import pandas as pd

def create_plot_directory():
    """创建保存图片的文件夹"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    plot_dir = os.path.join(results_dir, f"plots_{timestamp}")
    os.makedirs(plot_dir, exist_ok=True)
    return plot_dir

def save_plot(plt, filename, plot_dir):
    """保存图片的辅助函数"""
    plt.savefig(os.path.join(plot_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()

def plot_roc_curves(model_scores, strategy_name, plot_dir):
    """绘制ROC曲线"""
    plt.figure(figsize=(12, 8))
    plt.title(f'ROC Curves for {strategy_name} Sampling Strategy')
    
    for name, scores in model_scores[strategy_name].items():
        if 'roc_data' in scores:
            fpr, tpr = scores['roc_data']
            auc = scores['auc']
            plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='best')
    plt.grid(True)
    save_plot(plt, f'roc_curve_{strategy_name}.png', plot_dir)

def plot_pr_curves(model_scores, strategy_name, plot_dir):
    """绘制PR曲线"""
    plt.figure(figsize=(12, 8))
    plt.title(f'Precision-Recall Curves for {strategy_name} Sampling Strategy')
    
    for name, scores in model_scores[strategy_name].items():
        if 'pr_data' in scores:
            precision, recall = scores['pr_data']
            ap = scores['average_precision']
            plt.plot(recall, precision, label=f'{name} (AP = {ap:.2f})')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc='best')
    plt.grid(True)
    save_plot(plt, f'pr_curve_{strategy_name}.png', plot_dir)

def plot_shap_values(model, X_test, selected_features, plot_dir):
    """绘制SHAP值图"""
    # 确保X_test是DataFrame，如果不是则转换为DataFrame
    if not isinstance(X_test, pd.DataFrame):
        X_test = pd.DataFrame(X_test, columns=selected_features)
    
    try:
        # 首先尝试使用TreeExplainer（适用于树模型）
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
    except:
        try:
            # 如果TreeExplainer失败，尝试使用KernelExplainer
            def model_predict(x):
                if hasattr(model, 'predict_proba'):
                    return model.predict_proba(x)[:, 1]
                else:
                    return model.predict(x)
            
            # 使用KernelExplainer需要背景数据，这里使用X_test的均值作为背景
            background = shap.kmeans(X_test, 10)
            explainer = shap.KernelExplainer(model_predict, background)
            shap_values = explainer.shap_values(X_test)
        except Exception as e:
            print(f"无法计算SHAP值: {str(e)}")
            return
    
    # 如果shap_values是列表，取正类（类别1）的值
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # 取正类的SHAP值
    
    try:
        # 绘制SHAP值分布图
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_test, show=False)
        plt.title('SHAP值分布图 (正类)')
        plt.xlabel('SHAP值')
        plt.ylabel('特征')
        plt.tight_layout()
        save_plot(plt, 'shap_values.png', plot_dir)
        
        # 绘制特征重要性排序图
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
        plt.title('特征重要性排序 (正类)')
        plt.xlabel('平均|SHAP值|')
        plt.ylabel('特征')
        plt.tight_layout()
        save_plot(plt, 'feature_importance.png', plot_dir)
        
        # 绘制每个特征的依赖图
        print("\n绘制特征依赖图...")
        for feature in selected_features:
            plt.figure(figsize=(10, 6))
            try:
                # 确保X_test是DataFrame
                X_test_df = pd.DataFrame(X_test, columns=selected_features)
                
                # 获取特征值并处理异常值
                feature_values = X_test_df[feature].values
                feature_values = np.clip(feature_values, 
                                       np.percentile(feature_values, 1), 
                                       np.percentile(feature_values, 99))
                
                # 获取特征索引
                feature_idx = selected_features.index(feature)
                
                # 确保shap_values是二维数组
                if len(shap_values.shape) == 1:
                    shap_values = shap_values.reshape(-1, 1)
                
                # 绘制SHAP依赖图
                shap.dependence_plot(
                    feature_idx,
                    shap_values,
                    X_test_df,
                    feature_names=selected_features,
                    show=False,
                    interaction_index=None  # 不显示交互特征
                )
                
                # 添加趋势线
                x = feature_values
                y = shap_values[:, feature_idx]
                
                # 使用多项式拟合趋势线
                z = np.polyfit(x, y, 3)  # 3次多项式拟合
                p = np.poly1d(z)
                
                # 生成平滑的趋势线点
                x_smooth = np.linspace(x.min(), x.max(), 100)
                y_smooth = p(x_smooth)
                
                # 绘制趋势线
                plt.plot(x_smooth, y_smooth, 'r-', linewidth=2, label='趋势线')
                
                # 优化图形显示
                plt.title(f'{feature}的SHAP依赖图 (正类)', fontsize=14, pad=20)
                plt.xlabel(feature, fontsize=12)
                plt.ylabel('SHAP值', fontsize=12)
                plt.legend(loc='best')
                plt.grid(True, alpha=0.3)
                
                # 调整布局
                plt.tight_layout()
                
                # 保存图片
                save_plot(plt, f'shap_dependence_{feature}.png', plot_dir)
            except Exception as e:
                print(f"警告：无法绘制 {feature} 的依赖图: {str(e)}")
                plt.close()
                
    except Exception as e:
        print(f"绘制SHAP图时出错: {str(e)}")

def plot_calibration_curve(y_test, y_pred_proba, plot_dir):
    """绘制校准曲线"""
    prob_true, prob_pred = calibration_curve(y_test, y_pred_proba, n_bins=10)
    
    plt.figure(figsize=(10, 6))
    plt.plot(prob_pred, prob_true, marker='o', label='预测概率')
    plt.plot([0, 1], [0, 1], linestyle='--', label='理想校准线')
    plt.xlabel('预测概率')
    plt.ylabel('实际概率')
    plt.title('校准曲线')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    save_plot(plt, 'calibration_curve.png', plot_dir)

def plot_decision_curve(y_test, y_pred_proba, plot_dir):
    """绘制决策曲线"""
    # 确保输入是numpy数组
    if isinstance(y_test, pd.Series):
        y_test = y_test.values
    if isinstance(y_pred_proba, pd.Series):
        y_pred_proba = y_pred_proba.values
    
    # 确保y_pred_proba是一维数组
    if len(y_pred_proba.shape) > 1:
        y_pred_proba = y_pred_proba.squeeze()
    
    thresholds = np.linspace(0, 0.99, 100)
    net_benefit = []
    
    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        tp = np.sum((y_pred == 1) & (y_test == 1))
        fp = np.sum((y_pred == 1) & (y_test == 0))
        n = len(y_test)
        if threshold == 0:
            net_benefit.append(0)
        else:
            net_benefit.append((tp - (threshold/(1-threshold))*fp) / n)
    
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, net_benefit, label='模型', linewidth=2)
    plt.plot([0, 1], [0, 0], '--', label='不治疗')
    plt.plot([0, 1], [0, -1], '--', label='全部治疗')
    plt.xlabel('阈值概率')
    plt.ylabel('净收益')
    plt.title('决策曲线分析')
    plt.legend(loc='lower left')
    plt.grid(True, alpha=0.3)
    save_plot(plt, 'decision_curve.png', plot_dir)

def plot_bootstrap_distributions(bootstrap_metrics, plot_dir):
    """绘制Bootstrap结果的分布图"""
    plt.figure(figsize=(15, 10))
    for i, (metric, values) in enumerate(bootstrap_metrics.items()):
        plt.subplot(2, 3, i+1)
        plt.hist(values, bins=30, alpha=0.7)
        plt.axvline(np.mean(values), color='r', linestyle='dashed', linewidth=1)
        plt.axvline(np.percentile(values, 2.5), color='g', linestyle='dashed', linewidth=1)
        plt.axvline(np.percentile(values, 97.5), color='g', linestyle='dashed', linewidth=1)
        plt.title(f'{metric} Distribution')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
    plt.tight_layout()
    save_plot(plt, 'bootstrap_distributions.png', plot_dir) 