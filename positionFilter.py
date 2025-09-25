import pandas as pd
import numpy as np
from scipy.spatial import cKDTree  # KDTree 加速
from PMV import calculatePMV
import os

class position_choice():

    def __init__(self,filenamePre,savedir):
        self.fileNamePre = filenamePre

        self.a = pd.read_csv(filenamePre)

        self.columns_to_compare = ['Points:0', 'Points:1', 'Points:2']
        self.savedir = savedir
        fn = filenamePre.replace('.csv', '').split('_')
        self.name = fn[-5:]

    def fuzzy_merge_kdtree(self,b,tol_min,tol_max):
        """
        用 KDTree 近似匹配 df1 和 df2
        """
        b = pd.read_csv(b)
        bb = b[self.columns_to_compare].values
        tree = cKDTree(b[self.columns_to_compare].values)  # 构建 KDTree
        matched_indices = []

        # 查询 df1 每个点在 df2 中的最近邻，返回索引和距离
        distances, indices = tree.query(self.a[self.columns_to_compare].values, distance_upper_bound=tol_max)

        for i, (dist, idx) in enumerate(zip(distances, indices)):
            # 在 (tol_min, tol_max) 范围内才算匹配
            if tol_min < dist < tol_max:
                matched_indices.append(i)

        return self.a.iloc[matched_indices].reset_index(drop=True)
    
    def save_file(self,position_path,save_path,tol_min,tol_max):
        merged = self.fuzzy_merge_kdtree(position_path,tol_min,tol_max)
        merged.to_csv(save_path,index=False)

    def cal_concentration(self,file):
        point = ['Mass_fraction_of_co2']
        co2 = np.array(pd.read_csv(file)[point])
        return np.average(co2)
    
    def cal_pmv(self,file):
        cp = calculatePMV(file)
        pmvfile=cp.savedata()
        point = ['PMV']
        pmv = np.array(pd.read_csv(file)[point])
        pmv_ratio = np.sum(abs(pmv)<0.5)/len(pmv)
        return pmv_ratio
        
    def post_calculate(self):

        positionBreathe = 'comparedata/exportbreatheAreaCO2.csv'
        positionHuman = 'comparedata/position_cook.csv'
        positionOutlet = 'comparedata/exportoutletCO2.csv'

        breathe_position = os.path.join(self.savedir, 'breatheArea_'+self.fileNamePre.split('\\')[-1])
        human_position = os.path.join(self.savedir, 'human_'+self.fileNamePre.split('\\')[-1])
        outlet_position = os.path.join(self.savedir, 'outlet_'+self.fileNamePre.split('\\')[-1])

        self.save_file(positionBreathe,breathe_position,-0.01,0.01)
        self.save_file(positionHuman,human_position,0.01,0.03)
        self.save_file(positionOutlet,outlet_position,-0.01,0.01)

        cal_breathe = abs(self.cal_concentration(breathe_position))
        cal_outlet = abs(self.cal_concentration(outlet_position))

        cal_PMV = self.cal_pmv(human_position)

        ventilation_eff = cal_outlet/cal_breathe

        print('VE:'+str(ventilation_eff)+'PMV:'+str(cal_PMV))

        return ventilation_eff, cal_PMV

    # 加载两个 CSV 文件
    # a = pd.read_csv('AutoCFD/Workdata/Fluent_Python/EVADATA/new_DataSave_32_0.05_290_0.0_0.71.csv')
    # # a1 = pd.read_csv('AutoCFD/Workdata/Fluent_Python/EVADATA/new_DataSave_34_0.06_292_0.26_0.68.csv')
    # # a2 = pd.read_csv('AutoCFD/Workdata/Fluent_Python/EVADATA/new_DataSave_36_0.04_296_0.26_0.48.csv')

    # apre = pd.read_csv('predicted_snapshot_co2_32_0.05_290_0.0_0.71.csv')
    # # apre1 = pd.read_csv('comparedata/predicted_snapshot_co2_34_0.06_292_0.26_0.68.csv')
    # # apre2 = pd.read_csv('comparedata/predicted_snapshot_co2_36_0.04_296_0.26_0.48.csv')

    # b = pd.read_csv('comparedata/position_cook.csv')

    # columns_to_compare = ['Points:0', 'Points:1', 'Points:2']
    # name = ['32_0.05_290_0.0_0.71','34_0.06_292_0.26_0.68','36_0.04_296_0.26_0.48']

    # tolerance = 1e-2  # 容差



    # # ---------------- 真值数据 ----------------
    # nt = 0
    # for i in [a]:#, a1, a2]:
    #     merged = fuzzy_merge_kdtree(i, b, columns_to_compare, tolerance)
    #     merged.to_csv('comparedata/humankind_'+name[nt]+'_true.csv', index=False)
    #     print(f"共找到 {len(merged)} 条匹配的行，已保存至 humankind_{name[nt]}_true.csv")
    #     nt += 1

    # # ---------------- 预测数据 ----------------
    # nt = 0
    # for i in [apre]:#, apre1, apre2]:
    #     merged = fuzzy_merge_kdtree(i, b, columns_to_compare, tolerance)
    #     merged.to_csv('comparedata/humankind_'+name[nt]+'_pre.csv', index=False)
    #     print(f"共找到 {len(merged)} 条匹配的行，已保存至 humankind_{name[nt]}_pre.csv")
    #     nt += 1

if __name__ == "__main__":
    pc = position_choice(filenamePre='D:\\NextPaper\\code\\AutoCFD\\Workdata\\Fluent_Python\\NEWUSEDATA\\PMV_PMV_new_DataSave_26_0.07_290_0.26_0.68.csv',savedir='D:/NextPaper/code/AutoCFD/Workdata/Fluent_Python/NEWUSEDATA')
    pc.post_calculate()