import pandas as pd

# 读取两个Excel文件
df1 = pd.read_excel('TdP PT信号计算结果及比对情况最全最终整理2023.11.14word版.xlsx', sheet_name='Sheet1')
df2 = pd.read_excel('药物组合统计.xlsx', sheet_name='LW+CN药物组合')

# 定义一个函数来检查并填充表格2的数据
def fill_data(row):
    drug_combination = row['Drug Combination']
    drugs = drug_combination.split(' & ')
    
    # 查找表格1中匹配的行
    if len(drugs) == 2:
        match = df1[((df1['drug1'] == drugs[0]) & (df1['drug2'] == drugs[1])) | 
                    ((df1['drug1'] == drugs[1]) & (df1['drug2'] == drugs[0]))]
    else:
        match = pd.DataFrame()
    
    if not match.empty:
        # 填充Ω025
        omega025 = match.iloc[0]['Ω025']
        row['Ω025'] = '阳' if omega025 > 0 else '阴'
        
        # 填充X
        x = match.iloc[0]['X']
        row['X'] = '阳' if x > 2 else '阴'
        
        # 填充PRR、χ2、CRR
        prr = match.iloc[0]['PRR']
        chi2 = match.iloc[0]['χ2']
        crr = match.iloc[0]['CRR']
        row['PRR、χ2、CRR'] = '阳' if (prr > 2 and chi2 > 4 and crr > 2) else '阴'
        
        # 填充AM
        am = match.iloc[0]['AM']
        row['AM'] = '阳' if am > 0 else '阴'
        
        # 填充 Lexicomp® 和 Drugs.com®
        row['Lexicomp®'] = match.iloc[0]['Lexicomp®']
        row['Drugs.com®'] = match.iloc[0]['Drugs.com®']
    
    return row

# 应用函数填充数据
df2 = df2.apply(fill_data, axis=1)

# 保存到新的Excel文件
df2.to_excel('填充后的药物组合统计.xlsx', index=False)