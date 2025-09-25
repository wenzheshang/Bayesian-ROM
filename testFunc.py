# import os
# import csv
# import re

# def extract_params_from_filename(filename):
#     """
#     从文件名中提取最后四个参数
#     文件名格式: ..._param1_param2_param3_param4.csv
#     """
#     # 移除文件扩展名
#     name_without_ext = os.path.splitext(filename)[0]
    
#     # 使用正则表达式匹配最后四个由下划线分隔的部分
#     pattern = r'([^_]+)_([^_]+)_([^_]+)_([^_]+)$'
#     match = re.search(pattern, name_without_ext)
    
#     if match:
#         return match.groups()
#     return None

# def save_params_to_csv(directory, output_file):
#     """
#     从目录中的文件名提取参数并保存到CSV文件
#     """
#     # 收集所有参数
#     all_params = []
    
#     for filename in os.listdir(directory):
#         if filename.endswith('.csv'):
#             params = extract_params_from_filename(filename)
#             if params:
#                 all_params.append(params)
    
#     # 保存到CSV
#     with open(output_file, 'w', newline='') as csvfile:
#         writer = csv.writer(csvfile)
#         # 写入标题行
#         writer.writerow(['param1', 'param2', 'param3', 'param4'])
#         # 写入数据
#         writer.writerows(all_params)
    
#     print(f"成功提取 {len(all_params)} 个文件的参数并保存到 {output_file}")
#     return len(all_params)

# if __name__ == "__main__":
#     # 配置参数
#     target_directory = "AutoCFD/Workdata/Fluent_Python/NEWUSEDATA"  # 替换为包含CSV文件的目录
#     output_filename = "file_parameters.csv"    # 输出文件名
    
#     # 执行提取和保存
#     count = save_params_to_csv(target_directory, output_filename)
    
#     if count == 0:
#         print("警告: 未找到任何符合条件的CSV文件")

# import os
# from collections import defaultdict

# def find_duplicate_file_parts(directory):
#     # 存储每个最后四部分组合对应的文件名
#     part_dict = defaultdict(list)
    
#     # 遍历目录中的所有文件
#     for filename in os.listdir(directory):
#         # 检查是否为CSV文件（可根据实际扩展名调整）
#         if filename.endswith('.csv'):
#             # 去除文件扩展名
#             name_without_ext = os.path.splitext(filename)[0]
            
#             # 按'_'分割文件名
#             parts = name_without_ext.split('_')
            
#             # 确保文件名至少有4个部分（否则无法取最后四个）
#             if len(parts) >= 4:
#                 # 提取最后四个部分
#                 last_four = '_'.join(parts[-4:])
                
#                 # 存储到字典中
#                 part_dict[last_four].append(filename)
            
    
#     # 筛选出重复的组合
#     duplicates = {k: v for k, v in part_dict.items() if len(v) > 1}
    
#     return duplicates

# # 使用示例
# if __name__ == "__main__":
#     target_directory = 'AutoCFD/Workdata/Fluent_Python/NEWUSEDATA'  # 替换为实际目录路径
#     duplicate_files = find_duplicate_file_parts(target_directory)
    
#     if duplicate_files:
#         print("发现重复的文件名部分:")
#         for part, files in duplicate_files.items():
#             print(f"\n重复部分 '{part}' 出现在以下文件中:")
#             for file in files:
#                 print(f"  - {file}")
#     else:
#         print("未发现重复的文件名部分")
import numpy as np
import os
import pandas as pd
def load_snapshots(data_dir):
    T_snapshots, V_snapshots, Mass_snapshots, Vm_snapshots, BC = [], [], [], [], []
    coords = None

    for fname in os.listdir(data_dir):
        if not fname.endswith('.csv'):
            continue
        filepath = os.path.join(data_dir, fname)
        df = pd.read_csv(filepath)
        if coords is None:
            coords = df[["Points:0", "Points:1", "Points:2"]].values
        T_snapshots.append(df['Temperature'].values)
        Mass_snapshots.append((df['Mass_fraction_of_co2'].values)**0.5)
        V_snapshots.append(df[['Velocity:0', 'Velocity:1', 'Velocity:2']].values)
        Vm_snapshots.append((df['Velocity'].values)**0.5)

        # 提取边界条件
        parts = fname.replace('.csv', '').split('_')
        BC.append([float(parts[5]), float(parts[6]), float(parts[7]), float(parts[8])])

    return np.array(BC)

def find_duplicate_rows(matrix):
    row_dict = {}
    duplicates = {}
    
    for idx, row in enumerate(matrix):
        row_tuple = tuple(row)
        if row_tuple in row_dict:
            first_occurrence = row_dict[row_tuple]
            if first_occurrence not in duplicates:
                duplicates[first_occurrence] = [first_occurrence, idx]
            else:
                duplicates[first_occurrence].append(idx)
        else:
            row_dict[row_tuple] = idx
    
    if not duplicates:
        print("没有重复行")
    else:
        for first_idx, dup_list in duplicates.items():
            print(f"重复行发现：行{first_idx}和{', '.join(f'行{i}' for i in dup_list[1:])}内容相同")

# 示例用法
if __name__ == "__main__":
    matrix = load_snapshots('D:/NextPaper/code/AutoCFD/Workdata/Fluent_Python/NEWUSEDATA')
    matrix = np.array(matrix)[:,[0,2,3]]
    find_duplicate_rows(matrix)

new_bc = [1,2,3,4]
print([new_bc[0],new_bc[2],new_bc[3]])
pmv=np.array([1,2,3,-4,-5,0.5,0.3,-0.2])
pmv_ratio = np.sum(abs(pmv)<0.5)/len(pmv)
print(pmv_ratio)

import pandas as pd
import numpy as np
from scipy.spatial import cKDTree  # KDTree 加速

# 加载两个 CSV 文件
a = pd.read_csv('AutoCFD/Workdata/Fluent_Python/EVADATA/new_DataSave_32_0.05_290_0.0_0.71.csv')
# a1 = pd.read_csv('AutoCFD/Workdata/Fluent_Python/EVADATA/new_DataSave_34_0.06_292_0.26_0.68.csv')
# a2 = pd.read_csv('AutoCFD/Workdata/Fluent_Python/EVADATA/new_DataSave_36_0.04_296_0.26_0.48.csv')

apre = pd.read_csv('predicted_snapshot_co2_32_0.05_290_0.0_0.71.csv')
# apre1 = pd.read_csv('comparedata/predicted_snapshot_co2_34_0.06_292_0.26_0.68.csv')
# apre2 = pd.read_csv('comparedata/predicted_snapshot_co2_36_0.04_296_0.26_0.48.csv')

b = pd.read_csv('comparedata/position_cook.csv')

columns_to_compare = ['Points:0', 'Points:1', 'Points:2']
name = ['32_0.05_290_0.0_0.71','34_0.06_292_0.26_0.68','36_0.04_296_0.26_0.48']

tolerance = 1e-2  # 容差

def fuzzy_merge_kdtree(df1, df2, cols, tol):
    """
    用 KDTree 近似匹配 df1 和 df2
    """
    tree = cKDTree(df2[cols].values)  # 构建 KDTree
    matched_indices = []

    # 查询 df1 每个点在 df2 中的最近邻，返回索引和距离
    distances, indices = tree.query(df1[cols].values, distance_upper_bound=tol)

    for i, (dist, idx) in enumerate(zip(distances, indices)):
        if dist != np.inf:  # 说明在 tol 范围内找到匹配
            matched_indices.append(i)

    return df1.iloc[matched_indices].reset_index(drop=True)

# ---------------- 真值数据 ----------------
nt = 0
for i in [a]:#, a1, a2]:
    merged = fuzzy_merge_kdtree(i, b, columns_to_compare, tolerance)
    merged.to_csv('comparedata/humankind_'+name[nt]+'_true.csv', index=False)
    print(f"共找到 {len(merged)} 条匹配的行，已保存至 humankind_{name[nt]}_true.csv")
    nt += 1

# ---------------- 预测数据 ----------------
nt = 0
for i in [apre]:#, apre1, apre2]:
    merged = fuzzy_merge_kdtree(i, b, columns_to_compare, tolerance)
    merged.to_csv('comparedata/humankind_'+name[nt]+'_pre.csv', index=False)
    print(f"共找到 {len(merged)} 条匹配的行，已保存至 humankind_{name[nt]}_pre.csv")
    nt += 1


import pandas as pd

# 加载两个 CSV 文件
a = pd.read_csv('AutoCFD/Workdata/Fluent_Python/EVADATA/new_DataSave_32_0.05_290_0.0_0.71.csv')#pd.read_csv('comparedata/predicted_snapshot_co2.csv')#
a1 = pd.read_csv('AutoCFD/Workdata/Fluent_Python/EVADATA/new_DataSave_34_0.06_292_0.26_0.68.csv')
a2 = pd.read_csv('AutoCFD/Workdata/Fluent_Python/EVADATA/new_DataSave_36_0.04_296_0.26_0.48.csv')

apre = pd.read_csv('comparedata/predicted_snapshot_co2_32_0.05_290_0.0_0.71.csv')
apre1 = pd.read_csv('comparedata/predicted_snapshot_co2_34_0.06_292_0.26_0.68.csv')
apre2 = pd.read_csv('comparedata/predicted_snapshot_co2_36_0.04_296_0.26_0.48.csv')

b = pd.read_csv('comparedata/position_cook.csv')
#b1 = pd.read_csv('comparedata/hengjiemian.csv')

# 只保留我们关心的三个坐标列
columns_to_compare = ['Points:0', 'Points:1', 'Points:2']
name = ['32_0.05_290_0.0_0.71','34_0.06_292_0.26_0.68','36_0.04_296_0.26_0.48']
nt = 0
for i in[a,a1,a2]:
    a_subset = i[columns_to_compare]
    b_subset = b[columns_to_compare]
    #b1_subset = b1[columns_to_compare]

    # 合并两个 DataFrame，找出重复（即两者都存在）的行
    merged = pd.merge(i, b_subset.drop_duplicates(), on=columns_to_compare, how='inner')
    #merged1 = pd.merge(i, b1_subset.drop_duplicates(), on=columns_to_compare, how='inner')


    # 保存结果为 c.csv
    merged.to_csv('comparedata/humankind_'+name[nt]+'_true.csv', index=False)
    #merged1.to_csv('comparedata/hengjiemian_'+name[nt]+'_true.csv', index=False)
    nt = nt+1
    print(f"共找到 {len(merged)} 条匹配的行，已保存至 095jiemian_true.csv")
    #print(f"共找到 {len(merged1)} 条匹配的行，已保存至 hengjiemian_true.csv")


nt = 0
for i in[apre,apre1,apre2]:
    a_subset = i[columns_to_compare]
    b_subset = b[columns_to_compare]
    #b1_subset = b1[columns_to_compare]

    # 合并两个 DataFrame，找出重复（即两者都存在）的行
    merged = pd.merge(i, b_subset.drop_duplicates(), on=columns_to_compare, how='inner')
    #merged1 = pd.merge(i, b1_subset.drop_duplicates(), on=columns_to_compare, how='inner')


    # 保存结果为 c.csv
    merged.to_csv('comparedata/humankind_'+name[nt]+'_pre.csv', index=False)
    #merged1.to_csv('comparedata/hengjiemian_'+name[nt]+'_pre.csv', index=False)
    nt = nt+1
    print(f"共找到 {len(merged)} 条匹配的行，已保存至 095jiemian_pre.csv")
    #print(f"共找到 {len(merged1)} 条匹配的行，已保存至 hengjiemian_pre.csv")


# import csv
# import numpy as np
# import math
# def evaluate(individual):
#         cfdv = (int(individual[0])-3)*0.01 + 0.05167969  #(0.04976563+0.05359375)/2±0.02
#         cfdt = (int(individual[1])-1)*2 + 290 #290K~298K (294K ± 4K)
#         cfda = [((math.pi/2)-(int(individual[2])-1)*(15*math.pi/180))/math.pi*180,((int(individual[3]))*(15*math.pi/180))/math.pi*180,((math.pi/2)-(int(individual[3]))*(15*math.pi/180))/math.pi*180]
#         cfdvector = [round(math.cos(cfda[0]),7),round(math.sin(cfda[0])*math.cos(cfda[1]),7),round(math.sin(cfda[0])*math.cos(cfda[2]),7)]
#         print(round(cfdv,2),round(cfdt,2),round(cfda[0],2),round(cfda[1],2))
#         #contam, Energy = CFD_simu(velocity=cfdv, temperature = cfdt)           
#         return #, Energy #flow
    
    
# filename='testInput.csv'
# data = []

# with open(filename, newline='', encoding='utf-8') as csvfile:
#     reader = csv.reader(csvfile)
#     for row in reader:
#         data.append(row)

# individual = np.array(data[1:])

# for i in range(0,len(individual)):
#     evaluate(individual[i])
# import h5py

# def inspect_cgns_structure(file_path):
#     def print_structure(name, obj):
#         if isinstance(obj, h5py.Dataset):
#             print(f"[Dataset] {name} -> shape: {obj.shape}, dtype: {obj.dtype}")
#         elif isinstance(obj, h5py.Group):
#             print(f"[Group]   {name}")

#     with h5py.File(file_path, 'r') as f:
#         print(f"Inspecting CGNS file: {file_path}")
#         f.visititems(print_structure)

# # 示例调用
# if __name__ == "__main__":
#     # 请替换成你自己的文件路径
#     file_path = "your_snapshot.cgns"
#     inspect_cgns_structure(file_path)
# import pandas as pd

# # 文件路径
# file1_path = 'new_input.csv'
# file2_path = 'testInput.csv'
# output_path = 'final_input.csv'

# # 读取文件，只关注前四列
# df1 = pd.read_csv(file1_path, usecols=[0, 1, 2, 3])
# df2 = pd.read_csv(file2_path, usecols=[0, 1, 2, 3])

# # 为了便于比较，创建唯一标识的字符串列（也可以用 tuple）
# df1['key'] = df1.astype(str).agg('_'.join, axis=1)
# df2['key'] = df2.astype(str).agg('_'.join, axis=1)

# # 找出 df1 中不在 df2 中的 key
# diff_keys = ~df1['key'].isin(df2['key'])

# # 重新读取 file1 的完整数据（如果你要保留整行）
# full_df1 = pd.read_csv(file1_path)
# filtered_df = full_df1[diff_keys]

# # 保存到新的文件
# filtered_df.to_csv(output_path, index=False)

# print(f"筛选完成，不在 file2 中的组合已保存至: {output_path}")


from cmath import sqrt
import csv,os
from datetime import datetime
import pandas as pd
import math
import numpy as np

data_dir = "AutoCFD/Workdata/Fluent_Python/NEWUSEDATA"
for fname in os.listdir(data_dir):
    if not fname.endswith('.csv'):
        continue
    df = pd.read_csv(os.path.join(data_dir, fname))
    df['Velocity']=np.sqrt(np.square(np.array(df['Velocity:0']))+np.square(np.array(df['Velocity:1']))+np.square(np.array(df['Velocity:2'])))
    df.to_csv(data_dir+'/new_'+fname,index=False)

# def power_func(x, a, b):
#     """
#     幂函数形式：y = a * x^b
#     :param x: 输入变量
#     :param a: 系数
#     :param b: 幂指数
#     :return: 计算结果
#     """
#     return a *(0.71**(1/3))* x ** b

# def fit_power_function(x_data, y_data):
#     """
#     拟合幂函数：y = a * x^b
#     :param x_data: 自变量数组（需 > 0）
#     :param y_data: 因变量数组（需 > 0）
#     :return: 拟合参数 (a, b)
#     """
#     try:
#         # 初始猜测值 [a, b]
#         initial_guess = [1.0, 1.0]
#         params, _ = curve_fit(power_func, x_data, y_data, p0=initial_guess, maxfev=10000)
#         return params
#     except RuntimeError as e:
#         print("拟合失败：", e)
#         return None, None

# def plot_fit(x_data, y_data, a, b):
#     """
#     绘制拟合曲线和原始数据点
#     """
#     x_fit = np.linspace(min(x_data), max(x_data), 200)
#     y_fit = power_func(x_fit, a, b)

#     plt.figure(figsize=(8, 5))
#     plt.scatter(x_data, y_data, label='原始数据', color='blue')
#     plt.plot(x_fit, y_fit, 'r-', label=f"拟合函数: y = {a:.2f} * 0.71^0.333 * x^{b:.2f}")
#     plt.xlabel('x')
#     plt.ylabel('y')
#     plt.title('幂函数拟合：$y = a \cdot 0.71^0.333 \cdot x^b$')
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()
#     plt.show()

# # 示例数据（确保 x 和 y > 0）
# x = np.array([9100,12660,17670,22790])
# y = np.array([57,72,98,117])

# # 拟合并绘图
# a_fit, b_fit = fit_power_function(x, y)
# if a_fit is not None:
#     print(f"拟合结果：y = {a_fit:.4f} * x^{b_fit:.4f}")
#     plot_fit(x, y, a_fit, b_fit)
