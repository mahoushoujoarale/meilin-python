import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from imblearn.ensemble import BalancedRandomForestClassifier, RUSBoostClassifier

class NeuralNetwork(nn.Module):
    def __init__(self, input_size):
        super(NeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(input_size, 256)
        self.layer2 = nn.Linear(256, 128)
        self.layer3 = nn.Linear(128, 64)
        self.layer4 = nn.Linear(64, 32)
        self.layer5 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU(0.1)
        self.dropout = nn.Dropout(0.5)
        self.batch_norm1 = nn.BatchNorm1d(256)
        self.batch_norm2 = nn.BatchNorm1d(128)
        self.batch_norm3 = nn.BatchNorm1d(64)
        self.batch_norm4 = nn.BatchNorm1d(32)
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.batch_norm1(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)
        
        x = self.layer2(x)
        x = self.batch_norm2(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)
        
        x = self.layer3(x)
        x = self.batch_norm3(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)
        
        x = self.layer4(x)
        x = self.batch_norm4(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)
        
        x = self.layer5(x)
        return torch.sigmoid(x)

class AttentionNeuralNetwork(nn.Module):
    def __init__(self, input_size):
        super(AttentionNeuralNetwork, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.Tanh(),
            nn.Linear(64, input_size),
            nn.Softmax(dim=1)
        )
        self.layer1 = nn.Linear(input_size, 256)
        self.layer2 = nn.Linear(256, 128)
        self.layer3 = nn.Linear(128, 64)
        self.layer4 = nn.Linear(64, 32)
        self.layer5 = nn.Linear(32, 1)
        self.leaky_relu = nn.LeakyReLU(0.1)
        self.dropout = nn.Dropout(0.5)
        self.batch_norm1 = nn.BatchNorm1d(256)
        self.batch_norm2 = nn.BatchNorm1d(128)
        self.batch_norm3 = nn.BatchNorm1d(64)
        self.batch_norm4 = nn.BatchNorm1d(32)
        
    def forward(self, x):
        attention_weights = self.attention(x)
        x = x * attention_weights
        
        x = self.layer1(x)
        x = self.batch_norm1(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)
        
        x = self.layer2(x)
        x = self.batch_norm2(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)
        
        x = self.layer3(x)
        x = self.batch_norm3(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)
        
        x = self.layer4(x)
        x = self.batch_norm4(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)
        
        x = self.layer5(x)
        return torch.sigmoid(x)

def get_models(input_size, y_train):
    """获取所有模型"""
    return {
        # 传统模型
        "Logistic Regression": LogisticRegression(class_weight='balanced'),
        "Random Forest": RandomForestClassifier(class_weight='balanced'),
        "Gradient Boosting": GradientBoostingClassifier(),
        "XGBoost": XGBClassifier(eval_metric='logloss', scale_pos_weight=sum(y_train == 0) / sum(y_train == 1)),
        "CatBoost": CatBoostClassifier(silent=True, class_weights=[1, sum(y_train == 0) / sum(y_train == 1)]),
        "LightGBM": LGBMClassifier(verbose=-1, class_weight='balanced'),
        
        # 专门的不平衡数据模型
        "Balanced Random Forest": BalancedRandomForestClassifier(random_state=42),
        "RUSBoost": RUSBoostClassifier(random_state=42),
        
        # 神经网络模型
        "Neural Network": NeuralNetwork(input_size=input_size),
        "Attention Neural Network": AttentionNeuralNetwork(input_size=input_size)
    } 