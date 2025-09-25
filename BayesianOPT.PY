from __future__ import annotations
import os
import math
import time
import pathlib
import numpy as np
import pandas as pd

from typing import Dict, Tuple
from ConfigSpace import Configuration, ConfigurationSpace, Float, Categorical

from smac import HyperparameterOptimizationFacade as HPOFacade
from smac import Scenario

from positionFilter import position_choice
from PODGalerkinFull import predictFlow
from lib.Counter_run_time import CallingParameter


# === 数据读取函数 ===
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
        BC.append([float(parts[5]), float(parts[6]), float(parts[7]), float(parts[8])])

    return (
        np.array(coords),
        np.clip(np.array(T_snapshots), 290, 313),
        np.array(Mass_snapshots),
        np.array(V_snapshots),
        np.clip(np.array(Vm_snapshots), 0, 10),
        np.array(BC),
    )


@CallingParameter
def CFD_simu(**kwargs):
    pre = predictFlow(**kwargs)
    filename = pre.main()
    pc = position_choice(filenamePre=filename, savedir=kwargs["work_path"])
    ve, pmv = pc.post_calculate()
    return ve, pmv


# === 全局结果列表 ===
results = []

# === 多目标目标函数 ===
def target_function(config: Configuration, seed: int = 0) -> Dict[str, float]:
    cfdv = config["x_velocity"]
    cfdt = config["x_temperature"]
    a1   = config["x_angle1"]
    a2   = config["x_angle2"]

    # 映射到物理参数
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

    # 三个目标
    outputs = {
        "neg_ve": -ve,
        "pmv_abs": -pmv,
        "cfdv": cfdv,
    }

    # 保存输入 + 输出到全局列表
    results.append({
        "x_velocity": cfdv,
        "x_temperature": cfdt,
        "x_angle1": a1,
        "x_angle2": a2,
        **outputs,
    })

    return outputs

if __name__ == "__main__":
    now_time = time.strftime("%Y-%m-%d_%H-%M", time.localtime())
    cur_path = os.path.abspath(os.path.dirname(__file__))
    WORKPATH = pathlib.Path(cur_path + "/Workdata/" + now_time)
    os.makedirs(WORKPATH, exist_ok=True)

    COORDSINPUT, TSNAPINPUT, MASSSNAPINPUT, VSNAPINPUT, VMSNAPINPUT, BCINPUT = load_snapshots("D:/NextPaper/code/AutoCFD/Workdata/Fluent_Python/NEWUSEDATA")

    # 配置空间
    cs = ConfigurationSpace()
    cs = ConfigurationSpace({
        "x_velocity": Float("x_velocity", (0.03, 0.07)),  # m/s
        "x_temperature": Float("x_temperature", (290, 298)),  # K
        "x_angle1": Categorical("x_angle1", [1, 2]),
        "x_angle2": Categorical("x_angle2", [1, 2, 3, 4, 5]),
    })

    # Scenario
    scenario = Scenario(
        configspace=cs,
        deterministic=True,
        n_trials=120,
        objectives=["neg_ve", "pmv_abs", "cfdv"],
    )

    # Facade with multi-objective
    smac = HPOFacade(
        scenario=scenario,
        target_function=target_function,
        multi_objective_algorithm=HPOFacade.get_multi_objective_algorithm(
            scenario,
            objective_weights=[1, 1, 10],  # 三个目标权重可调整
        ),
        overwrite=True,
    )

    incumbents = smac.optimize()

    print("Pareto front (incumbents):")
    for inc in incumbents:
        cost = smac.validate(inc)
        print(inc, " -> ", cost)

    # 保存结果
    # 保存 Pareto 前沿
    rows = []
    for inc in incumbents:
        cost = smac.validate(inc)  # 返回 np.ndarray，长度=3
        cost_dict = {name: float(val) for name, val in zip(scenario.objectives, cost)}
        row = {**dict(inc), **cost_dict}
        rows.append(row)

    pareto_df = pd.DataFrame(rows)
    pareto_df.to_csv(os.path.join(WORKPATH, "smac_pareto.csv"), index=False)

    # 保存所有评估过的配置
    all_df = pd.DataFrame(results)
    all_df.to_csv(os.path.join(WORKPATH, "smac_all_results.csv"), index=False)

    print("Pareto front saved to smac_pareto.csv")
    print("All evaluated configs saved to smac_all_results.csv")

