from __future__ import annotations
import os
import math
import time
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import Dict, List, Any
from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, CategoricalHyperparameter

# SMAC imports: 兼容不同版本的导入位置
try:
    from smac.facade import HyperparameterOptimizationFacade as HPOFacade
    from smac.scenario import Scenario
    from smac.callback import Callback
except Exception:
    # 备用导入（某些旧版本有不同路径）
    from smac import HyperparameterOptimizationFacade as HPOFacade  # type: ignore
    from smac import Scenario  # type: ignore
    from smac.callback import Callback  # type: ignore

from pymoo.indicators.hv import HV

# === CFD 相关模块（保持你原有模块） ===
from positionFilter import position_choice
from PODGalerkinFull import predictFlow
from lib.Counter_run_time import CallingParameter

# ----------------- 数据读取函数 -----------------
def load_snapshots(data_dir: str):
    T_snapshots, V_snapshots, Mass_snapshots, Vm_snapshots, BC = [], [], [], [], []
    coords = None
    for fname in os.listdir(data_dir):
        if not fname.endswith(".csv"):
            continue
        filepath = os.path.join(data_dir, fname)
        df = pd.read_csv(filepath)
        if coords is None:
            coords = df[["Points:0", "Points:1", "Points:2"]].values
        T_snapshots.append(df["Temperature"].values)
        Mass_snapshots.append((df["Mass_fraction_of_co2"].values) ** 0.5)
        V_snapshots.append(df[["Velocity:0", "Velocity:1", "Velocity:2"]].values)
        Vm_snapshots.append((df["Velocity"].values) ** 0.5)

        parts = fname.replace(".csv", "").split("_")
        # 请确保文件名格式满足索引 5..8
        BC.append([float(parts[5]), float(parts[6]), float(parts[7]), float(parts[8])])

    return (
        np.array(coords),
        np.clip(np.array(T_snapshots), 290, 313),
        np.array(Mass_snapshots),
        np.array(V_snapshots),
        np.clip(np.array(Vm_snapshots), 0, 10),
        np.array(BC),
    )


# ----------------- CFD wrapper -----------------
@CallingParameter
def CFD_simu(**kwargs):
    pre = predictFlow(**kwargs)
    filename = pre.main()
    pc = position_choice(filenamePre=filename, savedir=kwargs["work_path"])
    ve, pmv = pc.post_calculate()
    return ve, pmv


# ----------------- 全局（注意并行时不可共享） -----------------
results: List[Dict[str, Any]] = []

# ----------------- Hypervolume 回调（兼容不同签名 + 实时绘图） -----------------
class HypervolumeCallback(Callback):
    def __init__(self, ref_point):
        super().__init__()
        self.ref_point = np.array(ref_point, dtype=float)
        self.hv_history: List[float] = []

        # 初始化绘图（交互模式）
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(6, 4))
        self.line, = self.ax.plot([], [], marker="o")
        self.ax.set_xlabel("Iteration")
        self.ax.set_ylabel("Hypervolume")
        self.ax.set_title("Hypervolume Convergence")
        self.ax.grid(True)
        # 记录 last x for plotting
        self._last_x = []

    def on_tell_end(self, *args, **kwargs):
        """
        兼容不同 SMAC 版本的回调调用：使用 *args/**kwargs 捕获所有可能的参数组合。
        首先尝试从全局 results 读取目标向量；若空则尝试从 smbo.runhistory 中读取（若可用）。
        """
        # 尝试从 kwargs 或 args 中找到 smbo（SMAC 的 smbo 对象通常是第一个位置参数）
        smbo = None
        if len(args) >= 1:
            smbo = args[0]
        else:
            smbo = kwargs.get("smbo", None)

        objs = None

        # 优先使用全局 results（简单且与你现有保存逻辑一致）
        try:
            if len(results) > 0:
                objs = np.array([[r["neg_ve"], r["pmv_abs"], r["cfdv"]] for r in results], dtype=float)
        except Exception:
            objs = None

        # 如果全局 results 为空，尝试从 smbo.runhistory 提取（对并行或其他调用更稳妥）
        if (objs is None or objs.size == 0) and smbo is not None:
            try:
                rh = getattr(smbo, "runhistory", None)
                if rh is None:
                    # 有些版本可能放在 smbo.runhistory_ 或 smbo.solver.runhistory 等，尝试宽松访问
                    rh = getattr(smbo, "runhistory_", None)
                if rh is None and hasattr(smbo, "get_runhistory"):
                    rh = smbo.get_runhistory()
                if rh is not None:
                    vals = []
                    data_dict = getattr(rh, "data", None)
                    if data_dict is None and hasattr(rh, "get_all_runs"):
                        # 保险尝试：不同版本 API 可能不同
                        try:
                            for run in rh.get_all_runs():
                                cost = getattr(run, "cost", None)
                                if cost is None:
                                    cost = getattr(run, "costs", None)
                                if cost is None:
                                    continue
                                if isinstance(cost, (list, tuple, np.ndarray)):
                                    vals.append([float(x) for x in cost])
                                else:
                                    vals.append([float(cost)])
                        except Exception:
                            data_dict = None
                    if data_dict:
                        # rh.data is dict: key->RunValue
                        for _k, rv in data_dict.items():
                            # RunValue 可能有属性 'cost' 或 'costs'
                            cost = getattr(rv, "cost", None)
                            if cost is None:
                                cost = getattr(rv, "costs", None)
                            if cost is None:
                                continue
                            if isinstance(cost, (list, tuple, np.ndarray)):
                                vals.append([float(x) for x in cost])
                            else:
                                vals.append([float(cost)])
                    if len(vals) > 0:
                        objs = np.array(vals, dtype=float)
            except Exception:
                objs = None

        # 若仍然没有 objs，则直接返回（不阻塞优化）
        if objs is None or objs.size == 0:
            return None

        # Ensure objs is 2D and has shape (_, 3) for our 3 objectives
        if objs.ndim != 2 or objs.shape[1] < 1:
            return None

        # 如果 objs 的目标数量不是 3（你的 scenario 是 3 目标），尝试取前 3 列或跳过
        if objs.shape[1] < 3:
            # 若确实只有一个目标（不太可能），则无法计算 3D HV；直接跳过
            return None
        if objs.shape[1] > 3:
            objs = objs[:, :3]

        # 计算 HV（尽量捕获异常）
        try:
            hv_indicator = HV(ref_point=self.ref_point)
            hv_value = float(hv_indicator(objs))
        except Exception as e:
            print("[HypervolumeCallback] HV calculation failed:", e)
            return None

        # 记录并打印
        self.hv_history.append(hv_value)
        print(f"[EHVI] Iter {len(self.hv_history)} -> HV={hv_value:.6f}")

        # 实时绘图更新
        try:
            x = np.arange(1, len(self.hv_history) + 1)
            y = np.array(self.hv_history)
            self.line.set_data(x, y)
            self.ax.relim()
            self.ax.autoscale_view()
            plt.draw()
            # pause 给 GUI 事件循环一点时间（若在 headless 环境可小到 0）
            plt.pause(0.01)
        except Exception:
            # 若绘图失败不影响优化
            pass

        return None


# ----------------- 目标函数（调用 CFD） -----------------
def target_function(config, seed: int = 0) -> Dict[str, float]:
    # 兼容 config 是 dict 或 ConfigSpace.Configuration
    cfdv = float(config["x_velocity"])
    cfdt = float(config["x_temperature"])
    a1 = int(config["x_angle1"])
    a2 = int(config["x_angle2"])

    # 映射角度（保留你原先公式）
    cfdv = round(cfdv, 3)
    cfdt = round(cfdt, 2)
    cfda = [
        (math.pi / 2) - (a1 - 1) * (15 * math.pi / 180),
        a2 * (15 * math.pi / 180),
        (math.pi / 2) - (a2 * 15 * math.pi / 180),
    ]
    cfdvector = [
        round(math.cos(cfda[0]), 2),
        round(math.sin(cfda[0]) * math.cos(cfda[1]), 2),
        round(math.sin(cfda[0]) * math.cos(cfda[2]), 2),
    ]
    cfda1, cfda2 = cfdvector[0], cfdvector[1]

    # 调用 CFD（确保全局变量在主程序中初始化）
    ve, pmv = CFD_simu(
        velocity=cfdv,
        temperature=cfdt,
        a1=cfda1,
        a2=cfda2,
        coords=COORDSINPUT,
        T_snaps=TSNAPINPUT,
        Mass_snaps=MASSSNAPINPUT,
        V_snaps=VSNAPINPUT,
        Vm_snaps=VMSNAPINPUT,
        BC=BCINPUT,
        work_path=WORKPATH,
    )

    outputs = {
        "neg_ve": float(-ve),
        "pmv_abs": float(-pmv),
        "cfdv": float(cfdv),
    }

    # 保存到全局 results（注意：并行时请改用 RunHistory）
    results.append({
        "x_velocity": cfdv,
        "x_temperature": cfdt,
        "x_angle1": a1,
        "x_angle2": a2,
        **outputs,
    })

    return outputs


# ----------------- 主程序 -----------------
if __name__ == "__main__":
    now_time = time.strftime("%Y-%m-%d_%H-%M", time.localtime())
    # __file__ 在交互式可能不存在，做回退
    if "__file__" in globals():
        cur_path = os.path.abspath(os.path.dirname(__file__))
    else:
        cur_path = os.path.abspath(os.getcwd())
    WORKPATH = pathlib.Path(cur_path) / "Workdata" / now_time
    os.makedirs(WORKPATH, exist_ok=True)

    # 读取快照数据（替换为你的目录）
    data_dir = "D:/NextPaper/code/AutoCFD/Workdata/Fluent_Python/NEWUSEDATA"
    COORDSINPUT, TSNAPINPUT, MASSSNAPINPUT, VSNAPINPUT, VMSNAPINPUT, BCINPUT = load_snapshots(data_dir)

    # 配置空间（使用 .add() 避免弃用警告）
    cs = ConfigurationSpace()
    cs.add(UniformFloatHyperparameter("x_velocity", lower=0.03, upper=0.07))
    cs.add(UniformFloatHyperparameter("x_temperature", lower=290.0, upper=298.0))
    cs.add(CategoricalHyperparameter("x_angle1", choices=[1, 2]))
    cs.add(CategoricalHyperparameter("x_angle2", choices=[2, 3, 4, 5]))

    # Scenario
    scenario = Scenario(
        configspace=cs,
        deterministic=True,
        n_trials=120,
        objectives=["neg_ve", "pmv_abs", "cfdv"],
    )

    # 创建回调（参考点请按你的目标范围调整）
    hv_callback = HypervolumeCallback(ref_point=[-0.01, -0.01, 1.0])

    # HPOFacade（注册回调）
    smac = HPOFacade(
        scenario=scenario,
        target_function=target_function,
        multi_objective_algorithm=HPOFacade.get_multi_objective_algorithm(
            scenario,
            objective_weights=[1, 1, 10],
        ),
        overwrite=True,
        callbacks=[hv_callback],
    )

    # 运行优化
    print("Starting optimization...")
    incumbents = smac.optimize()

    # 输出并保存 Pareto 前沿
    print("Pareto front (incumbents):")
    rows = []
    for inc in incumbents:
        cost = smac.validate(inc)
        # 转成 dict
        if isinstance(cost, (list, tuple, np.ndarray)):
            cost_dict = {name: float(val) for name, val in zip(scenario.objectives, cost)}
        else:
            cost_dict = {scenario.objectives[0]: float(cost)}
        print(inc, " -> ", cost_dict)
        rows.append({**dict(inc), **cost_dict})

    pareto_df = pd.DataFrame(rows)
    pareto_df.to_csv(os.path.join(WORKPATH, "smac_pareto.csv"), index=False)

    # 保存所有评估过的配置（注意：并行时全局 results 可能不完整）
    all_df = pd.DataFrame(results)
    all_df.to_csv(os.path.join(WORKPATH, "smac_all_results.csv"), index=False)

    # 保存 HV 历史
    hv_df = pd.DataFrame({"iteration": np.arange(1, len(hv_callback.hv_history) + 1),
                          "hypervolume": hv_callback.hv_history})
    hv_df.to_csv(os.path.join(WORKPATH, "hv_history.csv"), index=False)

    print("Pareto front saved to smac_pareto.csv")
    print("All evaluated configs saved to smac_all_results.csv")
    print("Hypervolume history saved to hv_history.csv")

    # 关闭交互并显示最终图像（阻塞）
    try:
        plt.ioff()
        plt.show()
    except Exception:
        pass
