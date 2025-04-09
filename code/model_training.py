import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, classification_report, confusion_matrix, precision_recall_curve, average_precision_score, brier_score_loss
from scipy.stats import sem
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from datetime import datetime

class CustomDataset(Dataset):
    def __init__(self, X, y):
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        self.X = torch.FloatTensor(X.copy())
        self.y = torch.FloatTensor(y.copy())
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def train_neural_network(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for X, y in train_loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs.squeeze(), y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def evaluate_neural_network(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            loss = criterion(outputs.squeeze(), y)
            total_loss += loss.item()
            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    
    return total_loss / len(test_loader), np.array(all_preds), np.array(all_labels)

def calculate_auc_ci(y_true, y_scores, alpha=0.95):
    auc = roc_auc_score(y_true, y_scores)
    n1 = sum(y_true)
    n2 = len(y_true) - n1
    q1 = auc / (2 - auc)
    q2 = 2 * auc**2 / (1 + auc)
    auc_var = (auc * (1 - auc) + (n1 - 1) * (q1 - auc**2) + (n2 - 1) * (q2 - auc**2)) / (n1 * n2)
    auc_std = np.sqrt(auc_var)
    ci = auc + np.array([-1, 1]) * sem(y_scores) * 1.96
    return auc, ci[0], ci[1]

def evaluate_model(name, model, X_train, y_train, X_test, y_test, is_neural_net=False, device=None):
    if is_neural_net:
        model = model.to(device)
        
        train_dataset = CustomDataset(X_train, y_train)
        test_dataset = CustomDataset(X_test, y_test)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=64)
        
        criterion = nn.BCELoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
        
        n_epochs = 200
        best_loss = float('inf')
        patience = 15
        patience_counter = 0
        
        print(f"\n开始训练 {name}...")
        for epoch in range(n_epochs):
            train_loss = train_neural_network(model, train_loader, criterion, optimizer, device)
            test_loss, y_pred_proba, y_true = evaluate_neural_network(model, test_loader, criterion, device)
            
            scheduler.step(test_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{n_epochs}] - Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")
            
            if test_loss < best_loss:
                best_loss = test_loss
                patience_counter = 0
                torch.save(model.state_dict(), 'best_model.pth')
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch}")
                model.load_state_dict(torch.load('best_model.pth'))
                break
                
        y_pred = (y_pred_proba > 0.5).astype(int)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_true = y_test
    
    metrics_dict = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'auc': roc_auc_score(y_test, y_pred_proba),
        'y_pred_proba': y_pred_proba
    }
    
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    metrics_dict['specificity'] = tn / (tn + fp)
    
    auc, auc_low, auc_up = calculate_auc_ci(y_test, y_pred_proba)
    metrics_dict['auc_ci'] = (auc_low, auc_up)
    
    print(f"\n{name} 模型评估结果:")
    print("\n分类报告:")
    print(classification_report(y_test, y_pred))
    
    print("\n混淆矩阵:")
    print(confusion_matrix(y_test, y_pred))
    
    print("\n主要评估指标:")
    print(f"AUC (ROC): {metrics_dict['auc']:.4f} [95% CI: {auc_low:.4f}-{auc_up:.4f}]")
    print(f"Accuracy: {metrics_dict['accuracy']:.4f}")
    print(f"Sensitivity/Recall: {metrics_dict['recall']:.4f}")
    print(f"Specificity: {metrics_dict['specificity']:.4f}")
    print(f"Precision: {metrics_dict['precision']:.4f}")
    print(f"F1-score: {metrics_dict['f1']:.4f}")
    
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    
    metrics_dict['average_precision'] = average_precision_score(y_test, y_pred_proba)
    metrics_dict['brier_score'] = brier_score_loss(y_test, y_pred_proba)
    metrics_dict['roc_data'] = (fpr, tpr)
    metrics_dict['pr_data'] = (precision, recall)
    
    return metrics_dict 

def evaluate_all_models(models, X_train, y_train, X_test, y_test, sampling_strategies, device):
    """评估所有模型"""
    results_data = []
    model_scores = {}
    
    for strategy_name, sampler in sampling_strategies.items():
        print(f"\n{'='*50}")
        print(f"使用 {strategy_name} 采样策略:")
        print(f"{'='*50}")
        
        # 应用采样策略
        X_train_res, y_train_res = sampler.fit_resample(X_train, y_train)
        
        # 打印采样后的数据分布
        print(f"\n采样后的数据分布:")
        print(f"正样本数量: {sum(y_train_res == 1)}")
        print(f"负样本数量: {sum(y_train_res == 0)}")
        print(f"正样本比例: {sum(y_train_res == 1) / len(y_train_res):.2%}")
        
        # 训练和评估每个模型
        strategy_scores = {}
        for name, model in models.items():
            is_neural_net = "Neural Network" in name
            print(f"\n{'-'*30}")
            print(f"训练模型: {name}")
            print(f"{'-'*30}")
            
            scores = evaluate_model(name, model, X_train_res, y_train_res, X_test, y_test, is_neural_net, device)
            strategy_scores[name] = scores
            
            # 将评估指标添加到结果列表
            results_data.append({
                '采样策略': strategy_name,
                '模型': name,
                'ROC-AUC': scores['auc'],
                'ROC-AUC 95%CI': f"{scores['auc_ci'][0]:.4f}-{scores['auc_ci'][1]:.4f}",
                'PR-AUC': scores['average_precision'],
                'Accuracy': scores['accuracy'],
                'Sensitivity': scores['recall'],
                'Specificity': scores['specificity'],
                'Precision': scores['precision'],
                'F1-score': scores['f1'],
                'Brier Score': scores['brier_score']
            })
        
        # 将策略的模型分数添加到主字典
        model_scores[strategy_name] = strategy_scores
    
    return results_data, model_scores

def find_best_model(model_scores, models, sampling_strategies, X_train, y_train, X_test, y_test, device):
    """找出并评估最佳模型"""
    best_score = -1
    best_model_name = None
    best_strategy_name = None
    
    # 确保y_test是numpy数组
    if isinstance(y_test, pd.Series):
        y_test = y_test.values
    
    # 遍历所有策略和模型，找到最佳组合
    for strategy_name, strategy_scores in model_scores.items():
        for model_name, scores in strategy_scores.items():
            # 计算综合评分
            # 1. 区分度指标 (Discrimination)
            discrimination_score = scores['auc'] * 0.4  # AUC权重40%
            
            # 2. 校准度指标 (Calibration)
            calibration_score = (1 - scores['brier_score']) * 0.2  # Brier分数权重20%
            
            # 3. 临床实用性指标 (Clinical Utility)
            # 使用决策曲线分析中的净收益
            thresholds = np.linspace(0, 0.99, 100)
            net_benefits = []
            y_pred_proba = scores['y_pred_proba']
            if isinstance(y_pred_proba, pd.Series):
                y_pred_proba = y_pred_proba.values
                
            for threshold in thresholds:
                y_pred = (y_pred_proba >= threshold).astype(int)
                tp = np.sum((y_pred == 1) & (y_test == 1))
                fp = np.sum((y_pred == 1) & (y_test == 0))
                n = len(y_test)
                if threshold == 0:
                    net_benefits.append(0)
                else:
                    net_benefits.append((tp - (threshold/(1-threshold))*fp) / n)
            clinical_utility_score = np.max(net_benefits) * 0.2  # 最大净收益权重20%
            
            # 4. 稳定性指标 (Stability)
            # 使用AUC的95%置信区间宽度作为稳定性指标
            auc_ci_width = scores['auc_ci'][1] - scores['auc_ci'][0]
            stability_score = (1 - auc_ci_width) * 0.2  # 置信区间宽度权重20%
            
            # 计算总分
            current_score = (
                discrimination_score + 
                calibration_score + 
                clinical_utility_score + 
                stability_score
            )
            
            if current_score > best_score:
                best_score = current_score
                best_model_name = model_name
                best_strategy_name = strategy_name
    
    print(f"\n{'='*50}")
    print(f"所有策略中的最佳模型: {best_strategy_name}-{best_model_name}")
    print(f"综合评分: {best_score:.4f}")
    print(f"{'='*50}")
    
    # 获取最佳模型和对应的采样策略
    sampler = sampling_strategies[best_strategy_name]
    X_train_res, y_train_res = sampler.fit_resample(X_train, y_train)
    best_model = models[best_model_name]
    
    # 训练最佳模型
    if "Neural Network" in best_model_name:
        best_model = best_model.to(device)
        test_dataset = CustomDataset(X_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size=64)
        best_model.eval()
        with torch.no_grad():
            y_pred_proba = []
            for X, _ in test_loader:
                X = X.to(device)
                outputs = best_model(X)
                y_pred_proba.extend(outputs.cpu().numpy())
            y_pred_proba = np.array(y_pred_proba).squeeze()
    else:
        best_model.fit(X_train_res, y_train_res)
        y_pred_proba = best_model.predict_proba(X_test)[:, 1]
    
    return best_model, y_pred_proba, best_model_name 