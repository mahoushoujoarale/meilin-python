# 不需要处理的标识列
id_cols = ['序号', 'patient_SN']

# 定义二分类列（假设这些列是用1和2表示的二分类数据）
binary_cols = ['性别', '民族', '吸烟', '饮酒', '低蛋白血症', '过敏史', '冠心病', '高血压', '2型糖尿病', '慢阻肺', '肺部感染', '肺炎', '肝/肾功能异常', '肿瘤', '心房颤动', '心力衰竭', '气管插管', '输血', '利尿剂', '抗心律失常药', '胃肠用药', '支气管扩张药', '心悸', '不良反应']

# 定义基线输入文件
baseline_input_files = ['negative-baseline.xlsx', 'positive-baseline.xlsx']

# 定义基线表名
# baseline_sheet_names = ['基线阴性插补前', '基线阳性插补前']
baseline_sheet_names = ['Sheet1', 'Sheet1']

# 定义比较输入文件
comparison_input_files = ['negative-comparison.xlsx', 'positive-comparison.xlsx']

# 定义合并后基线输入文件
merged_baseline_input_file = ['baseline.xlsx']

# 定义合并后基线表名
merged_baseline_sheet_names = ['Sheet1']

# 定义合并后比较输入文件
merged_comparison_input_file = ['comparison.xlsx']

# 定义合并后比较表名
merged_comparison_sheet_names = ['Sheet1']

# 定义比较表名
# comparison_sheet_names = ['阴性跑模型插补前', '阳性跑模型插补前']
comparison_sheet_names = ['Sheet1', 'Sheet1']

# 定义输入文件
# input_files = baseline_input_files + comparison_input_files + merged_baseline_input_file + merged_comparison_input_file
input_files = merged_baseline_input_file

# 定义表名
# sheet_names = baseline_sheet_names + comparison_sheet_names + merged_baseline_sheet_names + merged_comparison_sheet_names
sheet_names = merged_baseline_sheet_names