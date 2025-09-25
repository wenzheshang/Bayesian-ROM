# The aim of this code is organizing the NSGA-II with ROM model
# Find the pareto front of PMV and IAQ in the four-dimensional paramater space

from doctest import master
from subprocess import run
from positionFilter import position_choice
from PODGalerkinFull import predictFlow
import random
import numpy as np
import math
from deap import base, creator, tools, algorithms
import matplotlib.pyplot as plt
from lib.Counter_run_time import CallingParameter
import time
import pathlib
import os, sys
import subprocess
import pandas as pd
import itertools

import psutil

#记录一些不好记录的数据
dict_energy = {'energy':[]}
dict_outT = {'outT':[]}
dict_contam = {'cotam':[]}
dict_pmv = {'pmv':[]}
dict_ve = {'ve':[]}
dict_flow = {'flow':[]}

#可传参的调用脚本进行后处理的批处理
# def paraview_post_bat(save_path, open_path, **kwargs):

#     pvpythonPath = "F:/paraview/ParaView-5.9.1-Windows-Python3.8-msvc2017-64bit/ParaView-5.9.1-Windows-Python3.8-msvc2017-64bit/bin/pvpython.exe"
#     scriptPath = "D:/NextPaper/code/AutoCFD/view_sc.py"
#     scriptPath2 = "D:/NextPaper/code/AutoCFD/ParaSaveData.py"
#     dic = kwargs
#     name = ''
#     for n in dic:
#         name  = name + str(dic[n]).ljust(len(dic[n])+1)
#     save = str(save_path)
#     open = str(open_path)

#     #run(f'{pvpythonPath} {scriptPath} {save} {open} {name}', shell=True)

#     run(f'{pvpythonPath} {scriptPath2} {save} {open} {name}', shell=True)

#     return 


# === 数据读取 ===
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

    return np.array(coords), np.clip(np.array(T_snapshots),290,313), np.array(Mass_snapshots), np.array(V_snapshots), np.clip(np.array(Vm_snapshots),0,10), np.array(BC)

@CallingParameter
def CFD_simu(**kwargs):
    
    pre = predictFlow(**kwargs)
    filename = pre.main()

    pc = position_choice(filenamePre=filename,savedir=kwargs['work_path'])
    ve,pmv = pc.post_calculate()

    return ve, pmv #,massflow

def main(workPath):
    IND_size = 4
    # MIN = -10
    # MAX = 10
    random.seed(64)

    creator.create('FitnessMin', base.Fitness, weights = (1.0, 1.0, -10)) # Calculate PMV and Carbon Dioxide, with low ventilation
    creator.create('Individual', list, fitness = creator.FitnessMin)

    toolbox = base.Toolbox()

    def randomlist():
        return random.uniform(0,1)

    toolbox.register('attr_item', randomlist)
    toolbox.register('individual', tools.initRepeat, creator.Individual, toolbox.attr_item, n = IND_size)
    toolbox.register('population', tools.initRepeat, list, toolbox.individual)
    
    COORDSINPUT, TSNAPINPUT, MASSSNAPINPUT, VSNAPINPUT, VMSNAPINPUT, BCINPUT = load_snapshots('D:/NextPaper/code/AutoCFD/Workdata/Fluent_Python/NEWUSEDATA')
    WORKPATH = workPath

    def evaluate(individual):
        # cfdv = individual[0]*0.4 + 0.367 #1.07#0.77#  #0.367m/s~2.367m/s  (1.367m/s ± 1m/s)
        # cfdt = individual[1]*2 + 290.35 #295.35#294.35# #290.35K~300.35K (295.35K ± 5K)
        # cfda1 = individual[2]*2 + 0.15
        # cfda2 = individual[3]*5 + 0.05

        cfdv = round((((float(individual[0])-0.5)*4)*0.01 + 0.05167969),3)  #(0.04976563+0.05359375)/2±0.02
        cfdt = round(((float(individual[1]))*4*2 + 290),2) #290K~298K (294K ± 4K)

        if individual[2]>=0.5:
            individual[2]=2
        else:
            individual[2]=1

        if individual[3]>=0.8:
            individual[3]=5
        elif 0.8>individual[3]>=0.6:
            individual[3]=4
        elif 0.6>individual[3]>=0.4:
            individual[3]=3
        elif 0.4>individual[3]>=0.2:
            individual[3]=2
        elif 0.2>individual[3]>=0:
            individual[3]=1

        cfda = [(math.pi/2)-(int(individual[2])-1)*(15*math.pi/180),(int(individual[3]))*(15*math.pi/180),(math.pi/2)-(int(individual[3]))*(15*math.pi/180)]
        cfdvector = [round(math.cos(cfda[0]),7),round(math.sin(cfda[0])*math.cos(cfda[1]),7),round(math.sin(cfda[0])*math.cos(cfda[2]),7)]
        cfda1 = cfdvector[0]
        cfda2 = cfdvector[1]

        ve, humanPMV = CFD_simu(velocity=cfdv, temperature = cfdt, a1 = cfda1, a2 = cfda2, coords = COORDSINPUT,T_snaps = TSNAPINPUT,
                          Mass_snaps = MASSSNAPINPUT, V_snaps = VSNAPINPUT, Vm_snaps = VMSNAPINPUT, BC = BCINPUT,work_path = WORKPATH)
        dict_ve['ve'].append(ve)
        dict_pmv['pmv'].append(humanPMV)
        #contam, Energy = CFD_simu(velocity=cfdv, temperature = cfdt)           
        return ve,humanPMV,cfdv #flow

    toolbox.register("evaluate", evaluate)
    toolbox.register('mate', tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    toolbox.register("select", tools.selNSGA2)

    # toolbox.decorate("mate", history.decorator)
    # toolbox.decorate("mutate", history.decorator)

    # toolbox.decorate("mate", decorate.checkBounds(MIN, MAX))
    # toolbox.decorate("mutate", decorate.checkBounds(MIN, MAX))

    mstats = tools.Statistics(key=lambda ind: ind.fitness.values)
    mstats.register('avg', np.mean, axis = 0)
    mstats.register('std', np.std, axis = 0)
    mstats.register('min', np.min, axis = 0)
    mstats.register('max', np.max, axis = 0)

    CXPB = 0.5
    MUTPB = 0.5
    NGEN = 6
    MU = 10
    LAMBDA = 30

    pop = toolbox.population(n=MU)

    # history.update(pop)

    hof = tools.ParetoFront()

    algorithms.eaMuPlusLambda(pop, toolbox, MU, LAMBDA, CXPB, MUTPB, NGEN, mstats,
                              halloffame=hof)

    # graph = networkx.DiGraph(history.genealogy_tree)
    # graph = graph.reverse()     # Make the graph top-down
    # colors = [toolbox.evaluate(history.genealogy_history[i])[0] for i in graph]
    # networkx.draw(graph, node_color=colors)
    # plt.savefig(os.path.join(workPath,'history.svg'), dpi=900, format = 'svg')

    return pop, mstats, hof

if __name__=='__main__':
 ##################################################################################
    now_time = time.strftime('%Y-%m-%d_%H-%M', time.localtime())
    cur_path =  os.path.abspath(os.path.dirname(__file__))
    root_path = cur_path
    workPath = pathlib.Path(root_path+"/Workdata/"+now_time)

    folder1 = os.path.exists(workPath)

    if not folder1:
        os.makedirs(pathlib.Path(root_path+"/Workdata/"+now_time))
    #main()
    (p, sta, hof) = main(workPath)
    i = 0
    gen = []
    #fit_flow = []
    fit_contam = []
    fit_energy = []
    fit_velocity = []

    dataframeVT = pd.DataFrame({'Velocity':CFD_simu.velocity, 'inletT':CFD_simu.temperature, 'Angle1':CFD_simu.a1, 'Angle2':CFD_simu.a2,
                                 'PMV':dict_pmv['pmv'], 'Ventilation Efficiency':dict_ve['ve']})
    dataframeVT.to_csv(os.path.join(workPath, 'iterationDataLog.csv'))

    for ind in hof:
        i += 1
        #fit_flow += [ind.fitness.values[0]]
        fit_contam += [ind.fitness.values[0]]
        fit_energy += [ind.fitness.values[1]]
        fit_velocity += [ind.fitness.values[2]]
        gen += [i]
    
    mydataframe_flow = pd.DataFrame({'fit_ve': fit_contam, 'fit_pmv': fit_energy, 'fit_velocity':fit_velocity})
    mydataframe_flow.to_csv(os.path.join(workPath,'fitLog.csv'))

    fig, ax = plt.subplots(2, 1,figsize=(8,12), gridspec_kw={'height_ratios': [1,1]})
    ax1 = ax[0]
    line1 = ax1.plot(fit_energy, fit_contam, "b-", label="CFD Paerto front1")
    ax1.set_xlabel("PMV")
    ax1.set_ylabel("Ventilation Efficiency", color="b")
    for tl in ax1.get_yticklabels():
        tl.set_color("b")

    lns = line1
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc="center right")
 ################################################################################
    plt.show()