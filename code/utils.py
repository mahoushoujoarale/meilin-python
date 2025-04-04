import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime
import torch
import os
from model_training import CustomDataset
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from torch.utils.data import DataLoader

def setup_device():
    """设置设备"""
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    return device

def ensure_results_dir():
    """确保results目录存在"""
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    return results_dir

def generate_baseline_table(df, selected_features):
    """生成基线特征表格"""
    results_data = []
    
    for feature in selected_features:
        total_stats = df[feature].describe()
        group_stats = df.groupby('不良反应')[feature].describe()
        
        stat, p_value = stats.mannwhitneyu(
            df[df['不良反应'] == 0][feature],
            df[df['不良反应'] == 1][feature],
            alternative='two-sided'
        )
        
        results_data.append({
            '特征': feature,
            '总体 (Mean ± SD)': f"{total_stats['mean']:.2f} ± {total_stats['std']:.2f}",
            '无不良反应组 (Mean ± SD)': f"{group_stats.loc[0, 'mean']:.2f} ± {group_stats.loc[0, 'std']:.2f}",
            '不良反应组 (Mean ± SD)': f"{group_stats.loc[1, 'mean']:.2f} ± {group_stats.loc[1, 'std']:.2f}",
            'P值': f"{p_value:.3f}"
        })
    
    baseline_df = pd.DataFrame(results_data)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = ensure_results_dir()
    baseline_file = os.path.join(results_dir, f'baseline_characteristics_{timestamp}.xlsx')
    baseline_df.to_excel(baseline_file, index=False)
    print(f"\nBaseline characteristics have been saved to: {baseline_file}")
    
    return baseline_df

def save_results_to_excel(results_data, filename_prefix='model_evaluation_results'):
    """保存结果到Excel文件"""
    results_df = pd.DataFrame(results_data)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = ensure_results_dir()
    excel_file = os.path.join(results_dir, f'{filename_prefix}_{timestamp}.xlsx')
    results_df.to_excel(excel_file, sheet_name='评估结果', index=False)
    print(f"\nResults have been saved to: {excel_file}")

def perform_bootstrap_validation(model, X_test, y_test, n_bootstrap=1000, is_neural_net=False, device=None):
    """执行Bootstrap验证"""
    bootstrap_metrics = {
        'auc': [],
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'specificity': []
    }
    
    for i in range(n_bootstrap):
        indices = np.random.choice(len(X_test), size=len(X_test), replace=True)
        X_bootstrap = X_test.iloc[indices] if isinstance(X_test, pd.DataFrame) else X_test[indices]
        y_bootstrap = y_test.iloc[indices] if isinstance(y_test, pd.Series) else y_test[indices]
        
        if is_neural_net:
            bootstrap_dataset = CustomDataset(X_bootstrap, y_bootstrap)
            bootstrap_loader = DataLoader(bootstrap_dataset, batch_size=64)
            model.eval()
            with torch.no_grad():
                y_pred_proba = []
                for X, _ in bootstrap_loader:
                    X = X.to(device)
                    outputs = model(X)
                    y_pred_proba.extend(outputs.cpu().numpy())
                y_pred_proba = np.array(y_pred_proba).squeeze()
        else:
            y_pred_proba = model.predict_proba(X_bootstrap)[:, 1]
        
        y_pred = (y_pred_proba > 0.5).astype(int)
        bootstrap_metrics['auc'].append(roc_auc_score(y_bootstrap, y_pred_proba))
        bootstrap_metrics['accuracy'].append(accuracy_score(y_bootstrap, y_pred))
        bootstrap_metrics['precision'].append(precision_score(y_bootstrap, y_pred))
        bootstrap_metrics['recall'].append(recall_score(y_bootstrap, y_pred))
        bootstrap_metrics['f1'].append(f1_score(y_bootstrap, y_pred))
        tn, fp, fn, tp = confusion_matrix(y_bootstrap, y_pred).ravel()
        bootstrap_metrics['specificity'].append(tn / (tn + fp))
    
    return bootstrap_metrics

def print_bootstrap_results(bootstrap_metrics):
    """打印Bootstrap验证结果"""
    print("\n外部验证结果 (95%置信区间):")
    for metric, values in bootstrap_metrics.items():
        mean_value = np.mean(values)
        ci_low = np.percentile(values, 2.5)
        ci_high = np.percentile(values, 97.5)
        print(f"{metric}: {mean_value:.4f} [{ci_low:.4f}-{ci_high:.4f}]")

def save_external_validation_results(bootstrap_metrics, plot_dir):
    """保存外部验证结果到Excel文件"""
    external_validation_results = {
        'Metric': list(bootstrap_metrics.keys()),
        'Mean': [np.mean(values) for values in bootstrap_metrics.values()],
        '95% CI Lower': [np.percentile(values, 2.5) for values in bootstrap_metrics.values()],
        '95% CI Upper': [np.percentile(values, 97.5) for values in bootstrap_metrics.values()]
    }
    external_validation_df = pd.DataFrame(external_validation_results)
    external_validation_df.to_excel(os.path.join(plot_dir, 'external_validation_results.xlsx'), index=False) 