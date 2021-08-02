# -*- coding:utf-8 -*-
import numpy as np
import evaluation
import func_tools
import chemfile_transform
import load_data


import cantera as ct 
import matplotlib
import matplotlib.pyplot as plt 
import random
import sys, os
import json, time

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch



# 加载详细机理文件
input_file = '/home/wangzhiwei/combustion/code/chem.yaml'
all_species = ct.Species.listFromFile(input_file)
ref_phase = ct.Solution(thermo='ideal-gas', kinetics = 'gas', species=all_species)
all_reactions = ct.Reaction.listFromFile(input_file, ref_phase)

S = []

for s in all_species:
    S.append(s.name)

print(S)


# 采样点个数，简化机理个数
num_temperature, num_pressure = 9, 7

# 采样点取值范围
t_min, t_max, p_min, p_max = 1600, 1200, 0.1, 10

# 生成温度、压力
initial_temperature = np.linspace(t_min, t_max, num = num_temperature, endpoint = True)
initial_pressure = np.exp(np.linspace(np.log(p_min), np.log(p_max), num = num_pressure, endpoint = True))

# 保存采样点
sample_points = []
for i in range(num_temperature):
    for j in range(num_pressure):
        sample_points.append([initial_temperature[i], initial_pressure[j]])

epoch_num = 0
IDT, IDT_log = [], []

t = time.time()
for j in range(num_temperature):
    for i in range(num_pressure):
        t0 = time.time()
        # 实例化gas对象，必须每次重新实例化
        gas = ct.Solution(thermo='ideal-gas', kinetics='gas', species=all_species, reactions=all_reactions)
        gas.TPX = initial_temperature[j], initial_pressure[i] * ct.one_atm, 'H2:1, O2:1' 

        # 求点火延迟时间
        idt = evaluation.solve_idt(gas, 'H2', 1e-7)
        IDT.append(idt)
        IDT_log.append(np.log10(idt))
        time_cost = time.time()-t0
        print('TP index: %s, %s' %(i, j), 
         '\tTP:{:.2f} {:.2f}\t time cost:{:.3f}'.format(initial_temperature[j], initial_pressure[i], time_cost),
         '\tidt: %s' % (idt))
        epoch_num += 1
total_time = time.time()-t
print('cost %s s for generate data' % (total_time))

vector_len = len(all_reactions)


# 保存数据
np.savez('/home/wangzhiwei/combustion/idt_data/true_idt_data.npz', 
        IDT = IDT, 
        sample_points = sample_points,
        vector_len = vector_len)  


x = np.linspace(0, 63, 63)
# plt.scatter(x, F, s = 1)
plt.scatter(x, IDT_log, s=2)
plt.savefig('/home/wangzhiwei/combustion/code/true_idt_data.png')


'''
    这个反应能够输出所有组分反应过程中的摩尔分数变化
'''
def true_reaction():
    # 加载详细机理文件
    input_file = '/home/wangzhiwei/combustion/code/chem.yaml'
    all_species = ct.Species.listFromFile(input_file)
    gas = ct.Solution(thermo='ideal-gas', kinetics = 'gas', species=all_species)
    all_reactions = ct.Reaction.listFromFile(input_file, gas)

    # S是组分列表，N是所有组分在所有时刻的摩尔分数列表，Temp是温度列表
    S, N, Temp = [], [], []
    for s in all_species:
        S.append(s.name)
        N.append([])

    s_num = len(S)

    gas = ct.Solution(thermo='ideal-gas', kinetics='gas', species=all_species, reactions=all_reactions)
    gas.TPX = 1500, 3 * ct.one_atm, 'H2:2, O2:1' 

    r = ct.IdealGasReactor(gas)
    sim = ct.ReactorNet([r])

    times = []
    time = 0
    delta_t = 1e-7 # 设定时间步长
    # 主循环：时间推进
    for j in range(100):
        time += delta_t
        sim.advance(time) # 推进到t=time时间点
        times.append(time)
        Temp.append(r.T)
        if j % 10000 == 0:
            print(j)
        for i,s in enumerate(S):
            N[i].append(r.thermo[s].X[0])
    # 绘制点火延迟曲线以及所有物质的摩尔分数变化
    fig = plt.figure()
    plt.xlabel('time (s)')
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twinx()
    ax1.plot(times, Temp, color='r')
    for i in range(s_num):
        ax2.plot(times, N[i], color='b')
    plt.savefig('/home/wangzhiwei/combustion/code/all_species_frac')

    # 输出最终时刻各组分的摩尔分数
    for i in range(s_num):
        print(S[i],':',N[i][-1])





