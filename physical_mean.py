# -*- coding:utf-8 -*-

import numpy as np
import shutil
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import random
import sys, os
import time
import json
import cantera as ct
import multiprocessing as mp
from multiprocessing import Pool
import matplotlib
import matplotlib.pyplot as plt

from model_reduction import model_reduction, MyNet
from GenData_multiprocess import *


# 给定0-1向量，输出点火延迟和真实点火延迟之间的差异
def compare(x, tol = 1e-6):
    md = model_reduction('./chem.yaml')
    md.set_FO('H2','O2')
    md.sample_TP(num_temperature = 9, num_pressure = 7, t_min = 1200, t_max = 1600, p_min = 0.1, p_max = 10)
    md.set_time_step(delta_t = 1e-7)

    # 真实点火延迟时间
    true_Data = np.load('%s/true_idt_data.npz'% (md.true_data_path))
    true_idt  = np.log10(np.array(true_Data['IDT']))
    true_hrr  = np.log10(np.array(true_Data['HRR']))
    true_T    = np.log10(np.array(true_Data['T']))

    x_0 = 0.4 * np.random.rand(1,md.reactions_num) + 0.6
    x, predict_idt, predict_T, predict_hrr = md.solve_inverse(x_0, lr = 1e-2, t1 = 30, t2 = 30, t3 = 30, gpu_index = 1, 
                        iteration = 100000, lr_decay_step = 400, lr_decay_rate = 0.99)
    

    # 生成x对应的简化机理的点火延迟时间
    md.current_vector = x.tolist()
    
    # 加载详细机理文件
    ref_phase     = ct.Solution(thermo='ideal-gas', kinetics='gas', species = md.all_species)
    all_reactions = ct.Reaction.listFromFile(md.mechanism, ref_phase)
    
    # 获得简化机理涉及的反应
    reactions = vector2machanism(all_reactions, md.current_vector)
    # 获得简化机理涉及的组分
    species_names = {}
    for reaction in reactions:
        species_names.update(reaction.reactants)
        species_names.update(reaction.products)
    species = [ref_phase.species(name) for name in species_names]
    IDT, reduced_T, reduced_hrr = [], [], []
    for i in range(md.num_temperature):
        for j in range(md.num_pressure):
            gas = ct.Solution(thermo='ideal-gas', kinetics='gas', species = species, reactions = reactions)
            gas.TP = md.initial_temperature[i], md.initial_pressure[j] * ct.one_atm
            gas.set_equivalence_ratio(1, md.fuel, md.oxidizer)
            idt, T, hrr = solve_idt(gas, md.fuel, delta_t=1e-7)
            IDT.append(idt)
            reduced_T.append(T)
            reduced_hrr.append(hrr)
            print('i={}, j={}, idt={:4e}, T = {}, hrr = {}' .format(i, j ,idt, T, hrr))
    reduced_idt = np.log10(IDT)
    reduced_T   = np.log10(reduced_T)
    reduced_hrr = np.log10(reduced_hrr)
    
    # print the result
    print('reduced_IDT:\n', reduced_idt)
    print('true_IDT:\n', true_idt)
    # print('predict_IDT:\n', predict_idt)

    # plot the result
    multi_fig = plt.figure(figsize=(28.8,8))
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.95, top=0.9, wspace=0.25, hspace=0.25)
    ax1 = multi_fig.add_subplot(131)    # compare idt
    ref_x = [-5, -1]
    ref_y = [-5, -1]
    ax1.plot(ref_x, ref_y, 'r--', lw = 1)
    ax1.scatter(reduced_idt, true_idt,    c = 'b', s = 2, label = 'reduced_IDT vs true_IDT')
    ax1.scatter(predict_idt, true_idt,    c = 'g', s = 2, label = 'predict_IDT vs true_IDT')
    ax1.scatter(reduced_idt, predict_idt, c = 'k', s = 2, label = 'reduced_IDT vs predict_IDT')
    plt.legend(loc='upper left')

    ax2 = multi_fig.add_subplot(132)    # compare T
    ref_x = [3, 4]
    ref_y = [3, 4]
    ax2.plot(ref_x, ref_y, 'r--', lw = 1)
    ax2.scatter(reduced_T, true_T,    c = 'b', s = 2, label = 'reduced_T vs true_T')
    ax2.scatter(predict_T, true_T,    c = 'g', s = 2, label = 'predict_T vs true_T')
    ax2.scatter(reduced_T, predict_T, c = 'k', s = 2, label = 'reduced_T vs predict_T')
    plt.legend(loc='upper left')

    ax3 = multi_fig.add_subplot(133)    # compare hrr
    ref_x = [3, 11]
    ref_y = [3, 11]
    ax3.plot(ref_x, ref_y, 'r--', lw = 1)
    ax3.scatter(reduced_hrr, true_hrr,    c = 'b', s = 2, label = 'reduced_hrr vs true_hrr')
    ax3.scatter(predict_hrr, true_hrr,    c = 'g', s = 2, label = 'predict_hrr vs true_hrr')
    ax3.scatter(reduced_hrr, predict_hrr, c = 'k', s = 2, label = 'reduced_hrr vs predict_hrr')
    plt.legend(loc='upper left')

    plt.tight_layout()
    plt.savefig('./pic/compare.png')
    plt.close(multi_fig)


# 输出机理中涉及的组分和反应信息
def output_info():
    md = model_reduction('./chem.yaml')
    md.print_info()



'''
    这个反应能够输出所有组分反应过程中的摩尔分数变化
'''
def all_species_frac_plot(x):
    md = model_reduction('./chem.yaml')
    md.set_FO('H2','O2')
    md.sample_TP(num_temperature = 9, num_pressure = 7, t_min = 1200, t_max = 1600, p_min = 0.1, p_max = 10)
    md.set_time_step(delta_t = 1e-7)

    md.load_true_data()
    true_idt = np.array(md.true_idt_data).reshape(md.num_temperature, md.num_pressure)
    # 生成x对应的简化机理的点火延迟时间
    md.current_vector = x.tolist()
    
    # 加载详细机理文件
    all_species = ct.Species.listFromFile(md.mechanism)
    ref_phase = ct.Solution(thermo='ideal-gas', kinetics='gas', species=all_species)
    all_reactions = ct.Reaction.listFromFile(md.mechanism, ref_phase)
    
    # 获得简化机理涉及的反应
    reactions = vector2machanism(all_reactions, md.current_vector)
    # 获得简化机理涉及的组分
    species_names = {}
    for reaction in reactions:
        species_names.update(reaction.reactants)
        species_names.update(reaction.products)
    species = [ref_phase.species(name) for name in species_names]

    # time_str = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    # folder_name = './pic/all_species_frac%s' % (time_str)
    # if not os.path.exists(folder_name):
    #     os.mkdir(folder_name)
    
    for i in range(md.num_temperature):
        for j in range(md.num_pressure):
            print('ploting T = {:.0f}, P = {:.2f}'.format(md.initial_temperature[i], md.initial_pressure[j]))
            # S是组分列表，N是所有组分在所有时刻的摩尔分数列表，Temp是温度列表
            S, N, Temp = [], [], []
            N0, Temp0 = [], [] # 真实值列表
            for s in species:
                S.append(s.name)
                N.append([])
                N0.append([])

            gas = ct.Solution(thermo='ideal-gas', kinetics='gas', species = species, reactions = reactions)
            gas.TP = md.initial_temperature[i], md.initial_pressure[j] * ct.one_atm
            gas.set_equivalence_ratio(1, md.fuel, md.oxidizer)

            gas0 = ct.Solution(thermo='ideal-gas', kinetics='gas', species = all_species, reactions = all_reactions)
            gas0.TP = md.initial_temperature[i], md.initial_pressure[j] * ct.one_atm
            gas0.set_equivalence_ratio(1, md.fuel, md.oxidizer)


            r = ct.IdealGasReactor(gas)
            sim = ct.ReactorNet([r])

            r0 = ct.IdealGasReactor(gas0)
            sim0 = ct.ReactorNet([r0])

            times = []
            time = 0.0
            while time < 5 * true_idt[i, j]:
                time += md.delta_t
                sim.advance(time)
                sim0.advance(time)
                times.append(time)
                Temp.append(r.T)
                Temp0.append(r0.T)
                for t,s in enumerate(S):
                    N[t].append(r.thermo[s].X[0])
                    N0[t].append(r0.thermo[s].X[0])

            # 绘制点火延迟曲线以及所有物质的摩尔分数变化
            fig = plt.figure()
            plt.xlabel('time (s)')
            ax1 = fig.add_subplot(111)
            ax2 = ax1.twinx()
            ax1.plot(times, Temp, color='r',      label = 'T')
            ax2.plot(times, N[0], color='olive',  label = 'H')
            ax2.plot(times, N[1], color='b',      label = 'O2')
            ax2.plot(times, N[2], color='purple', label = 'O')
            ax2.plot(times, N[3], color='orange', label = 'OH')
            ax2.plot(times, N[4], color='g',      label = 'H2')
            ax2.plot(times, N[5], color='k',      label = 'H2O')
            ax2.plot(times, N[6], color='tan',    label = 'HO2')
            # ax2.plot(times, N[7], color='cyan',   label = 'H2O2')
            
            ax1.plot(times, Temp0, color='r',      linestyle='--', linewidth = 0.5)
            ax2.plot(times, N0[0], color='olive',  linestyle='--', linewidth = 0.5)
            ax2.plot(times, N0[1], color='b',      linestyle='--', linewidth = 0.5)
            ax2.plot(times, N0[2], color='purple', linestyle='--', linewidth = 0.5)
            ax2.plot(times, N0[3], color='orange', linestyle='--', linewidth = 0.5)
            ax2.plot(times, N0[4], color='g',      linestyle='--', linewidth = 0.5)
            ax2.plot(times, N0[5], color='k',      linestyle='--', linewidth = 0.5)
            ax2.plot(times, N0[6], color='tan',    linestyle='--', linewidth = 0.5)
            # ax2.plot(times, N0[7], color='cyan',   linestyle='--', linewidth = 0.5)
            plt.legend(loc='upper right')
            plt.tight_layout()
            # plt.savefig('./{}/{:.2f}T_{:.2f}p.png'.format(folder_name, md.initial_temperature[i], md.initial_pressure[j]))
            plt.savefig('./pic/all_species_frac20210724_124722/{:.2f}T_{:.2f}p.png'.format(md.initial_temperature[i], md.initial_pressure[j]))


if __name__ == "__main__":
    # x = np.array([ 5.1288e-01,  1.6729e+06, -6.5011e-01,  3.6777e-01,  4.6434e-01,
    #     -9.3620e-01,  8.1258e-01, -5.2013e-01,  9.7997e-01,  5.5200e-01,
    #     -5.9337e-01,  8.3043e-01, -6.8468e-01, -4.8487e-01,  8.3430e-01,
    #      4.5859e-01, -2.9619e+05,  6.6258e-01, -8.7754e-01,  6.6982e-01,
    #     -6.0729e-01, -3.7287e-01,  7.1977e-01,  5.7301e-01, -6.7387e-01,
    #      6.8442e-01, -8.0235e-01])
    # onehot_x = 1 * (1 / (1 + np.exp(-100 * x)) > 0.9)
    # print(np.sum(onehot_x))
    # print(27 - np.sum(onehot_x))
    # print(onehot_x)
    onehot_x = np.array([1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0])
    all_species_frac_plot(onehot_x)


    # x = np.ones(27)
    # x[0] = 1
    # x[1] = 1
    # x[2] = 0 # 重要反应，删去导致整体右移
    # x[3] = 1
    # x[4] = 0
    # x[5] = 0
    # x[8] = 0

    # '''稀有气体反应'''
    # x[6] = 0
    # x[7] = 0
    # x[9] = 0
    # x[10] = 0

    # x[11] = 0
    # x[12] = 0

    # '''生成HO2的反应'''
    # x[13] = 0
    # x[14] = 0 # 重要反应，删去导致个别点不准

    # '''消耗HO2的主导反应'''
    # x[15] = 1 # 删不得
    # '''消耗HO2的次级反应'''
    # x[16] = 0 # 删去导致有的结果不太准
    # x[17] = 0
    # x[18] = 0
    # x[19] = 0
    # x[20] = 0

    # '''消耗H2O2的反应'''
    # x[21] = 0
    # x[22] = 0
    # x[23] = 0 # 删去会导致有一个点的结果发生偏移
    # x[24] = 0
    # x[25] = 0
    # x[26] = 0