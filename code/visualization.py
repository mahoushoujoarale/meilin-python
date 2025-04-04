import matplotlib.pyplot as plt
import numpy as np
import shap
import os
from datetime import datetime
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.calibration import calibration_curve
from scipy import stats

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
    
    for name, scores in model_scores.items():
        fpr, tpr = scores['roc_data']
        plt.plot(fpr, tpr, label=f'{name} (AUC = {scores["auc"]:.2f})')
    
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
    
    for name, scores in model_scores.items():
        precision, recall = scores['pr_data']
        plt.plot(recall, precision, label=f'{name} (AP = {scores["average_precision"]:.2f})')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc='best')
    plt.grid(True)
    save_plot(plt, f'pr_curve_{strategy_name}.png', plot_dir)

def plot_shap_values(model, X_test, selected_features, plot_dir):
    """绘制SHAP值图"""
    explainer = shap.Explainer(model)
    shap_values = explainer.shap_values(X_test)
    
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_test, feature_names=selected_features, show=False)
    plt.title('SHAP值分布图')
    plt.xlabel('SHAP值')
    plt.ylabel('特征')
    plt.tight_layout()
    save_plot(plt, 'shap_values.png', plot_dir)
    
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_test, plot_type="bar", feature_names=selected_features, show=False)
    plt.title('特征重要性排序')
    plt.xlabel('|SHAP值|')
    plt.ylabel('特征')
    plt.tight_layout()
    save_plot(plt, 'feature_importance.png', plot_dir)

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