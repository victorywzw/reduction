# -*- coding:utf-8 -*-

from model_reduction import model_reduction, MyNet
from GenData_multiprocess import *
# 系统的包
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

def compare():
    md = model_reduction('./chem.yaml')
    md.set_FO('H2','O2')
    md.sample_TP(num_temperature = 9, num_pressure = 7, t_min = 1200, t_max = 1600, p_min = 0.1, p_max = 10)
    md.set_time_step(delta_t = 1e-7)

    # 真实点火延迟时间
    true_Data = np.load('%s/true_idt_data.npz'% (md.true_data_path))
    true_idt  = np.log10(np.array(true_Data['IDT']))
    true_T    = np.log10(np.array(true_Data['T']))

    x_0 = (np.random.rand(1, md.reactions_num) * 2 - 0.5) * 1/100
    x, predict_idt, predict_T, loss_his = md.solve_inverse2(x_0, lr = 10, t1 = 500, t2 = 1, t3 = 10, gpu_index = 1, 
                        iteration = 10000, lr_decay_step = 100, lr_decay_rate = 0.97)
    # x = np.array([ 0.8479, -0.5267, -0.1962,  1.5417,  0.1913, -0.1283, -0.1948, -0.2375,
    #      0.3822, -0.1261, -0.2019, -1.1984, -0.2091, -0.1378, -0.2267,  0.5575,
    #      0.1852, -0.1277, -0.1273, -0.1258, -0.1252,  4.1885, -0.2965,  0.1713,
    #     -0.1259, -0.1760, -0.1273])
    # x = 1 * (1 / (1 + np.exp(-100 * x)) > 0.9)
    # x = np.array([1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0])

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
    IDT, reduced_T = [], []
    for i in range(md.num_temperature):     # 计算reduced_ignition_data
        for j in range(md.num_pressure):
            gas = ct.Solution(thermo='ideal-gas', kinetics='gas', species = species, reactions = reactions)
            gas.TP = md.initial_temperature[i], md.initial_pressure[j] * ct.one_atm
            gas.set_equivalence_ratio(1, md.fuel, md.oxidizer)
            idt, T, hrr = solve_idt(gas, md.fuel, delta_t = md.delta_t)
            IDT.append(idt)
            reduced_T.append(T)
            print('i={}, j={}, idt={:4e}, T = {}, hrr = {}' .format(i, j ,idt, T, hrr))
    reduced_idt = np.log10(IDT)
    reduced_T   = np.log10(reduced_T)

    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    # 加载点火延迟的网络
    with open('%s/settings.json' % (md.model_idt_json_path),'r',encoding='utf8')as fp:
        json_data = json.load(fp)
    fp.close()
    model_idt = MyNet(json_data['input_dim'], json_data['hidden_units'], json_data['output_dim']).to(device)
    checkpoint = torch.load('%s/my_model%s.pth' % (md.model_idt_pth_path, json_data['train_index']), map_location = device)
    model_idt.load_state_dict(checkpoint['model'])
    # 加载最终温度的网络
    with open('%s/settings.json' % (md.model_T_json_path),'r',encoding='utf8')as fp:
        json_data = json.load(fp)
    fp.close()
    model_T = MyNet(json_data['input_dim'], json_data['hidden_units'], json_data['output_dim']).to(device)
    checkpoint = torch.load('%s/my_model%s.pth' % (md.model_T_pth_path, json_data['train_index']), map_location = device)
    model_T.load_state_dict(checkpoint['model'])
    x_01 = torch.FloatTensor(x)
    predict_idt = model_idt(x_01.to(device))
    predict_T   = model_T(x_01.to(device))
    predict_idt = predict_idt.cpu().detach().numpy()
    predict_T   = predict_T.cpu().detach().numpy()

    time_str = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    folder_name = './pic/all_species_frac%s' % (time_str)
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)
    
    # print the result
    print('reduced_IDT:\n', reduced_idt)
    print('true_IDT:\n', true_idt)
    print('predict_IDT:\n', predict_idt)

    # plt.plot(loss_his, 'r--',lw = 1)
    # plt.ylabel('loss_history')
    # plt.xlabel('iteration')
    # plt.xscale('log')
    # plt.yscale('log')
    # plt.savefig('./pic/inverse_loss.png')
    # plt.close()

    # plot the result
    multi_fig = plt.figure(figsize=(19.2, 8))
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.95, top=0.9, wspace=0.25, hspace=0.25)
    ax1 = multi_fig.add_subplot(121)    # compare idt
    ref_x0 = np.array([-7, -3])
    ref_y0 = np.array([-7, -3])
    ref_x1 = np.array([-6, -3])
    ref_y1 = np.array([-7, -4])
    ref_x2 = np.array([-7, -4])
    ref_y2 = np.array([-6, -3])
    ax1.plot(ref_x0, ref_y0, 'k--', lw = 0.5)
    ax1.plot(ref_x1, ref_y1, 'k--', lw = 0.5)
    ax1.plot(ref_x2, ref_y2, 'k--', lw = 0.5)
    ax1.scatter(reduced_idt, true_idt,    c = 'r', s = 4, label = 'reduced_IDT vs true_IDT')
    ax1.scatter(predict_idt, true_idt,    c = 'g', s = 4, label = 'predict_IDT vs true_IDT')
    ax1.scatter(reduced_idt, predict_idt, c = 'b', s = 4, label = 'reduced_IDT vs predict_IDT')
    plt.legend(loc='upper left')

    ax2 = multi_fig.add_subplot(122)    # compare T
    ref_x = [3, 4]
    ref_y = [3, 4]
    ax2.plot(ref_x, ref_y, 'r--', lw = 1)
    ax2.scatter(reduced_T, true_T,    c = 'b', s = 2, label = 'reduced_T vs true_T')
    ax2.scatter(predict_T, true_T,    c = 'g', s = 2, label = 'predict_T vs true_T')
    ax2.scatter(reduced_T, predict_T, c = 'k', s = 2, label = 'reduced_T vs predict_T')
    plt.legend(loc='upper left')

    plt.tight_layout()
    plt.savefig('%s/compare.png' % (folder_name))
    plt.close(multi_fig)

if __name__ == "__main__":
    compare()