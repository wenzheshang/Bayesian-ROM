# pareto_plot.py
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # 必须导入才能用 projection="3d"

def main():
    # 读取CSV文件
    iteration_df = pd.read_csv("Workdata/2025-09-16_16-11/smac_all_results.csv")
    fit_df = pd.read_csv("Workdata/2025-09-16_16-11/smac_pareto.csv")

    # 提取散点数据
    x = iteration_df["cfdv"]
    y = iteration_df["pmv_abs"]
    z = iteration_df["neg_ve"]

    # 提取帕累托前沿点
    fit_x = fit_df["cfdv"]
    fit_y = fit_df["pmv_abs"]
    fit_z = fit_df["neg_ve"]

    # 设置全局字体
    plt.rcParams["font.family"] = "Times New Roman"

    # 绘制三维图
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    # 绘制普通散点（浅黄色）
    ax.scatter(x, y, z, c="#477AAB", marker="o", s=25, alpha=0.6, label="Iteration Data")

    # 绘制帕累托前沿点（红粉色，单独绘制一次）
    ax.scatter(fit_x, fit_y, fit_z, c="#BD0200", marker="*", s=100, label="Pareto Points")

    # 使用三角剖分绘制曲面（蓝色）
    ax.plot_trisurf(fit_x, fit_y, fit_z, color="#F19149", alpha=0.3, edgecolor="none")

    # 调整视角
    # ax.view_init(elev=20, azim=120)  # elev=仰角, azim=方位角，可调

    # 调整Z轴到左侧
    ax.zaxis.set_tick_params(pad=10)  # 刻度和轴距离
    ax.zaxis._axinfo['juggled'] = (1,2,0)  # 强制Z轴放左边

    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        # 获取每个坐标轴的pane（坐标平面）
        pane = axis.pane
        # 设置pane的边框线型为实线，线宽可根据需要调整
        pane.set_linestyle('-')  # '-' 代表实线:cite[3]:cite[9]
        pane.set_linewidth(1.0)  # 设置边框线宽
        # 确保pane的边框是可见的
        # 也可以设置pane本身的填充颜色和透明度，使其不影响观察数据
        pane.set_facecolor('white')  # 设置平面背景色，若不需要可设alpha为0
        pane.set_alpha(0)  # 设置平面透明度，0为完全透明


    # 设置网格线为虚线
    ax.grid(True, linestyle="--", linewidth=0.6, color="gray", alpha=0.7)

    # 标签和标题
    ax.set_xlabel("Velocity (kg/s)", fontsize=14, labelpad=10)
    ax.set_ylabel("|PMV|<0.5 Ration (%)", fontsize=14, labelpad=10)
    ax.set_zlabel("Ventilation Efficiency", fontsize=14, labelpad=10)
    ax.set_title("Optimization Result", fontsize=16, pad=20)

    # 反转 PMV 轴
    ax.invert_yaxis()

    # 图例美化
    ax.legend(fontsize=12, loc="best")

    # 保存和显示
    plt.savefig("Result/Bayesian_2025-09-16_16-11.svg", dpi=900)
    plt.show()

def HVshow():
    hv_df = pd.read_csv("Workdata/2025-09-16_16-11/hv_history.csv")
    x = hv_df['iteration']
    y = hv_df['hypervolume']

    # 设置全局字体
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 12
    plt.rcParams["axes.linewidth"] = 1.2
    plt.rcParams["xtick.direction"] = "in"
    plt.rcParams["ytick.direction"] = "in"
    plt.rcParams["xtick.major.size"] = 6
    plt.rcParams["ytick.major.size"] = 6
    plt.rcParams["xtick.minor.size"] = 3
    plt.rcParams["ytick.minor.size"] = 3

    # 绘制图
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(x, y, color='#7E99F4', linewidth=2, linestyle='-')  # 曲线
    ax.scatter(x, y, color='#7E99F4', s=15, zorder=5)  # 散点

    # 设置标签和标题
    ax.set_xlabel("Iteration", fontsize=14, labelpad=10)
    ax.set_ylabel("Hypervolume", fontsize=14, labelpad=10)
    ax.set_title("Optimization Result", fontsize=16, pad=20)

    # 美化网格
    ax.grid(True, linestyle='-', alpha=0.5)

    # 设置刻度参数
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.tick_params(axis='both', which='minor', labelsize=10)

    # 设置 y 轴上限
    ax.set_ylim(top=0.5)

    # 保存高分辨率图片
    plt.tight_layout()
    plt.savefig("Result/HV_2025-09-16_16-11.svg", dpi=900)
    plt.show()


if __name__ == "__main__":
    main()
    HVshow()
