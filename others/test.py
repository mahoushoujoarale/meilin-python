import pandas as pd
from itertools import combinations

# 读取Excel文件
file_path = '非专业人士上报药物.xlsx'

# 定义处理函数
def process_sheet(sheet_name, output_file):
    # 读取指定工作表
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    
    # 根据case_id聚合，并对Drug name去重
    grouped = df.groupby('Case_id')['Drug name'].apply(lambda x: list(set(x))).reset_index()
    
    # 生成全组合
    results = []
    for _, row in grouped.iterrows():
        case_id = row['Case_id']
        drugs = row['Drug name']
        if len(drugs) == 1:
            results.append({'Case_id': case_id, 'Drug Combination': drugs[0]})
        else:
            # 生成所有两两组合
            combos = list(combinations(drugs, 2))
            for combo in combos:
                results.append({'Case_id': case_id, 'Drug Combination': f"{combo[0]} & {combo[1]}"})
    
    # 将结果转换为DataFrame
    result_df = pd.DataFrame(results)
    
    # 保存结果到新的Excel文件
    result_df.to_excel(output_file, index=False)
    print(f"工作表 '{sheet_name}' 的结果已保存到 {output_file}")

# 处理 LW药物组合 工作表
process_sheet('LW药物组合', 'LW_drug_combinations.xlsx')

# 处理 CN药物组合 工作表
process_sheet('CN药物组合', 'CN_drug_combinations.xlsx')