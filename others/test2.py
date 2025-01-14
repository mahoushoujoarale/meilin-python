import pandas as pd
from collections import defaultdict

# 读取Excel文件
df = pd.concat([pd.read_excel('LW_drug_combinations.xlsx'), pd.read_excel('CN_drug_combinations.xlsx')], ignore_index=True)

# 创建一个字典来存储组合及其计数
combination_counts = defaultdict(int)

# 遍历每一行
for index, row in df.iterrows():
    combination = row['Drug Combination']
    # 将组合按'&'分割并排序，以确保a & b和b & a被视为同一组合
    drugs = sorted([drug.strip() for drug in combination.split('&')])
    # 将排序后的药物重新组合成字符串
    sorted_combination = ' & '.join(drugs)
    # 更新组合计数
    combination_counts[sorted_combination] += 1

# 将字典转换为DataFrame
result_df = pd.DataFrame(list(combination_counts.items()), columns=['Drug Combination', 'Count'])

# 计算总数
total_count = result_df['Count'].sum()

# 计算百分比并添加到DataFrame中
result_df['Percentage'] = (result_df['Count'] / total_count) * 100

# 按照组合名称排序
result_df = result_df.sort_values(by='Drug Combination')

# 将结果写入新的Excel文件
result_df.to_excel('drug_combination_counts_with_percentage.xlsx', index=False)