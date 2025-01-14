import pandas as pd
import numpy as np
from fancyimpute import IterativeImputer
from sklearn.preprocessing import MinMaxScaler

# 读取数据
data = pd.read_excel('input.xlsx', sheet_name='Sheet1')

# 不需要处理的标识列
id_cols = ['序号', 'patient_SN']

# 定义二分类列（假设这些列是用1和2表示的二分类数据）
binary_cols = ['性别', '民族', '吸烟', '饮酒', '过敏史', '冠心病', '高血压', '2型糖尿病', '慢阻肺', '肺部感染', '肺炎', '肝/肾功能异常', '肿瘤', '心房颤动', '心力衰竭', '气管插管', '输血', '利尿剂', '抗心律失常药', '胃肠用药', '支气管扩张药', '心悸', '不良反应']

# 提取连续型数据列
continuous_cols = data.select_dtypes(include=[np.number]).columns.difference(id_cols).difference(binary_cols)

# 根据“不良反应”列的值对数据进行分组
groups = data.groupby('不良反应')

# 创建一个空列表，用于存储插补后的数据
imputed_data_list = []

# 对每一组分别进行插补和后续处理
for name, group in groups:
    # 多重插补处理（对所有数据，排除固定标识列）
    imputer = IterativeImputer(max_iter=10, random_state=0)  # 使用固定的随机种子
    group_imputed = group.copy()
    group_imputed[group_imputed.columns.difference(id_cols)] = imputer.fit_transform(group_imputed[group_imputed.columns.difference(id_cols)])

    # 将二分类数据规整为整数 0 和 1
    group_imputed[binary_cols] = group_imputed[binary_cols].round().astype(int) - 1

    # 归一化处理（仅对连续型数据）
    scaler = MinMaxScaler()  # 使用 MinMaxScaler 进行归一化
    group_imputed[continuous_cols] = scaler.fit_transform(group_imputed[continuous_cols])

    # 将处理后的数据添加到列表中
    imputed_data_list.append(group_imputed)

# 合并所有组的数据
data_imputed = pd.concat(imputed_data_list, ignore_index=True)

# 保存处理后的数据
output_filename = 'pretreatment.xlsx'
data_imputed.to_excel(output_filename, index=False)

print(f"插补处理完成并已保存为 '{output_filename}'")