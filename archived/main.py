import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from scipy.stats import sem
import shap

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Hiragino Sans GB']  # 使用 Hiragino Sans GB
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 读取数据集
df = pd.read_excel('pretreatment.xlsx')

# 选择特征和标签
# 这里指定你想要使用的特征列
# selected_features = ['钾离子(K+)', '间接胆红素(IBIL)', '尿酸(UA)', '碱性磷酸酶(ALP)', '氯离子(Cl-)', '钙离子(Ca2+)', '白细胞计数(WBC#)']
selected_features = ['肝/肾功能异常', '呼吸R', '尿酸(UA)', '白蛋白(ALB)', '碱性磷酸酶(ALP)', '氯离子(Cl-)', '钙离子(Ca2+)']
# selected_features = ['碱性磷酸酶(ALP)']
X = df[selected_features]
y = df['不良反应']

# 将标签从 [1, 2] 转换为 [0, 1]
y = np.where(y == 1, 0, 1)

# 对分类变量进行编码
X = pd.get_dummies(X)

# 划分数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 特征缩放
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 定义模型
models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(),
    "XGBoost": XGBClassifier(eval_metric='logloss'),
    "CatBoost": CatBoostClassifier(silent=True),
    "LightGBM": LGBMClassifier(verbose=-1),
}

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

def evaluate_model(name, model, X_train, y_train, X_test, y_test):
    # 训练模型
    model.fit(X_train, y_train)
    
    # 模型预测
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]  # 获取预测概率
    
    # 评估模型
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc, auc_low, auc_up = calculate_auc_ci(y_test, y_pred_proba)
    
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    specificity = tn / (tn + fp)
    
    print(f"{name} Accuracy (ACC): {accuracy}")
    print(f"{name} Precision (rf_precision): {precision}")
    print(f"{name} Recall (SEN/rf_recall): {recall}")
    print(f"{name} F1-score (rf_f1): {f1}")
    print(f"{name} AUC: {auc}")
    print(f"{name} AUC low: {auc_low}")
    print(f"{name} AUC up: {auc_up}")
    print(f"{name} Specificity (SPE): {specificity}")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("-" * 50)
    
    # 使用交叉验证来优化模型参数
    scores = cross_val_score(model, X_train, y_train, cv=10)
    print(f"{name} Cross-validated scores: {scores.mean()}")
    print("-" * 50)
    
    # 绘制 ROC 曲线
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')  # 绘制对角线
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='best')

# 训练和评估所有模型
plt.figure(figsize=(10, 8))
for name, model in models.items():
    evaluate_model(name, model, X_train, y_train, X_test, y_test)
plt.show()

# 计算 SHAP 值
explainer = shap.Explainer(models['CatBoost'])
shap_values = explainer.shap_values(X_test)

# 绘制 SHAP 值图
shap.summary_plot(shap_values, X_test, feature_names=selected_features)

# 绘制特征重要性图
shap.summary_plot(shap_values, X_test, plot_type="bar", feature_names=selected_features)