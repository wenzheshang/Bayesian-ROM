import math
import numpy as np
import os
import pandas as pd

class calculatePMV:
    def __init__(self,datadir):
        self.datadir = datadir
        
    def calculate_pmv(self,t_kelvin, V, p=101325, RH=0.5, M=69.78, W=0.0, icl=0.155):
        """
        ISO 7730 PMV/PPD calculation

        t_kelvin : 空气温度 (K)
        u, v, w  : 三个方向风速分量 (m/s)
        V        : 总速度(m/s)
        p        : 大气压 (Pa)
        RH       : 相对湿度 (0~1)
        M        : 代谢率 (W/m²)，1 met ≈ 58.15 W/m²
        W        : 外部功 (W/m²)，通常=0
        icl      : 服装热阻 (clo)，1 clo ≈ 0.155 m²K/W
        """

        # 转为 ℃
        t = t_kelvin - 273.15
        tr = t  # 假设平均辐射温度 = 空气温度

        # 有效风速
        speed = V#math.sqrt(u ** 2 + v ** 2 + w ** 2)
        if speed < 0.05:
            speed = 0.05

        # 饱和水蒸气压 (Pa)，Tetens公式
        psat = 610.5 * math.exp(17.2694 * t / (t + 237.3))
        pa = RH * psat

        # 衣服面积系数 fcl
        if icl < 0.078:
            fcl = 1.0 + 1.29 * icl
        else:
            fcl = 1.05 + 0.645 * icl

        # 初始 tcl 估计
        tcl1 = t + (35.5 - t) / (3.5 * icl + 0.1)
        tcl1 += 273.0  # 转换为 K

        # 迭代解 tcl
        while True:
            h_c_nat = 2.38 * abs(tcl1 - (t + 273.0)) ** 0.25
            h_c_for = 12.1 * math.sqrt(speed)
            hc = max(h_c_nat, h_c_for)

            tcl2 = (
                308.7 - 0.028 * (M - W)
                + icl * fcl * hc * (t + 273.0)
                + icl * 3.96e-8 * fcl * (tr + 273.0) ** 4
                - icl * 3.96e-8 * fcl * tcl1 ** 4
            ) / (1 + icl * fcl * hc)

            if abs(tcl1 - tcl2) <= 0.1:
                break
            tcl1 = tcl2

        tcl = tcl2 - 273.0

        # 热平衡各项 (W/m²)
        a = 3.96e-8 * fcl * ((tcl + 273.0) ** 4 - (tr + 273.0) ** 4)
        b = fcl * hc * (tcl - t)
        c = 0.00305 * (5733.0 - 6.99 * (M - W) - pa)
        d = 0.42 * (M - W - 58.15) if (M - W - 58.15) > 0 else 0.0
        e = 0.000017 * M * (5867.0 - pa)
        f = 0.0014 * M * (34.0 - t)

        # PMV
        L = M - W - (a + b + c + d + e + f)
        PMV = (0.303 * math.exp(-0.036 * M) + 0.028) * L
        PMV = max(-3, min(3, PMV))  # 限制范围

        # PPD
        PPD = 100 - 95 * math.exp(-0.03353 * PMV ** 4 - 0.2179 * PMV ** 2)

        return {"PMV": PMV, "PPD": PPD}


    def batch_calculate_pmv(self,data_points):
        results = []
        for i, point in enumerate(data_points):
            result = self.calculate_pmv(
                t_kelvin=np.clip(point['Temperature'],0,313.15),
                # u=point['Velocity:0'],
                # v=point['Velocity:1'],
                # w=point['Velocity:2'],
                V=point['Velocity'],
                p=101325,
                RH=point.get('RH', 0.05),
                M=point.get('M', 69.78),
                W=point.get('W', 0.0),
                icl=point.get('icl', 0.155)
            )
            results.append(result)
            #print(i)
        return results


    def run_from_csv(self,csv_path):
        df = pd.read_csv(csv_path)
        data_points = df.to_dict(orient='records')
        results = self.batch_calculate_pmv(data_points)
        df['PMV'] = [r['PMV'] for r in results]
        df['PPD'] = [r['PPD'] for r in results]
        return df


    def savedata(self):
    # 批量处理 CSV
        data_dir = self.datadir

        df_result = self.run_from_csv(data_dir)

        df_result.to_csv(os.path.join(data_dir), index=False)

# import math
# import numpy as np
# import os

# def calculate_pmv(
#     t_celsius, u, v, w, p, y_h2o,
#     M=69.78, W=0.0, icl=0.155
# ):
#     # 参数预处理
#     t = t_celsius-273.15  # ℃
#     tr = t  # 简化为同空气温度
#     speed = math.sqrt(u**2 + v**2 + w**2)
#     if speed < 0.05:
#         speed = 0.05

#     # 水蒸气分压
#     pa = y_h2o * p

#     # 衣服表面面积系数 fcl
#     if icl < 0.078:
#         fcl = 1.0 + 1.290 * icl
#     else:
#         fcl = 1.05 + 0.645 * icl

#     # 初始 tcl 估计
#     tcl1 = t + 273.0 + (35.5 - t) / (3.5 * icl + 0.92)

#     # 初始表面换热系数
#     hc = 12.1 * math.sqrt(speed)

#     #迭代求解 tcl
#     while True:
#         temp1 = 2.38 * abs(tcl1 - t - 273.0) ** 0.25
#         temp2 = 12.1 * math.sqrt(speed)
#         hc = max(temp1, temp2)

#         tcl2 = (
#             308.7 - 0.028 * (M - W)
#             + icl * fcl * hc * (t + 273.0)
#             + icl * 3.96e-8 * fcl * (tr + 273.0) ** 4
#             - icl * 3.96e-8 * fcl * tcl1 ** 4
#         ) / (1 + icl * fcl * hc)

#         if abs(tcl1 - tcl2) <= 0.1:
#             break
#         #print(tcl1-tcl2)
#         tcl1 = tcl2

#     tcl = tcl2 - 273.0

#     # 热损失分项
#     a = 3.96e-8 * fcl * ((tcl + 273.0) ** 4 - (tr + 273.0) ** 4)
#     b = fcl * hc * (tcl - t)
#     c = 0.00305 * (5733.0 - 6.99 * (M - W) - pa)
#     d = 0.42 * (M - W - 58.15) if (M - W - 58.15) > 0 else 0.0
#     e = 0.000017 * M * (5867.0 - pa)
#     f = 0.0014 * M * (34.0 - t)

#     # PMV 计算
#     L = M - W - (a + b + c + d + e + f)
#     PMV = (0.303 * math.exp(-0.036 * M) + 0.028) * L

#     return {
#         "PMV": PMV
#     }


# def batch_calculate_pmv(data_points):
#     """
#     data_points: list of dicts, each dict contains inputs for one point
#     Returns: list of result dicts with PMV, PPD, DR for each point
#     """
#     results = []
#     i=0
#     for point in data_points:
#         i=i+1
#         result = calculate_pmv(
#             t_celsius=np.clip(point['Temperature'],0,313.15),
#             u=point['Velocity:0'],
#             v=point['Velocity:1'],
#             w=point['Velocity:2'],
#             p=101325,
#             y_h2o=0.15,#point['y_h2o'],
#             M=point.get('M', 69.78),
#             W=point.get('W', 0.0),
#             icl=point.get('icl', 0.155)
#         )
#         results.append(result)
#         print(i)
#     return results

# import pandas as pd

# # from your_module import calculate_pmv_ppd_dr, batch_calculate_pmv_ppd_dr

# def run_from_csv(csv_path):
#     df = pd.read_csv(csv_path)

#     # 转为 dict list 供 batch 使用
#     data_points = df.to_dict(orient='records')

#     results = batch_calculate_pmv(data_points)

#     # 提取 PMV 列表
#     pmv_values = [res['PMV'] for res in results]

#     # 一次性赋值，大大提升速度
#     df['PMV'] = pmv_values

#     return df

# data_dir = "AutoCFD/Workdata/Fluent_Python/NEWUSEDATA"
# for fname in os.listdir(data_dir):
#     if not fname.endswith('.csv'):
#         continue
#     df_result = run_from_csv(os.path.join(data_dir, fname))
#     df_result.to_csv(data_dir+'/PMV_'+fname,index=False)