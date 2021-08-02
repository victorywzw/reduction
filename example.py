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

from model_reduction import model_reduction
from GenData_multiprocess import *


'''
    一个完整流程
    这里展示了如何用model_reduction class训练预测点火延迟的神经网络
    如果想训练预测最终火焰温度的神经网络只需将参数 mod = 0 改为 mod = 1 
'''
def full_procedure():
    md = model_reduction('./gri.yaml')  # 实例化，此时在当前代码目录下会产生文件夹
    md.set_FO('CH4','O2')               # 设置燃料和氧化剂
    md.sample_TP(num_temperature = 9, num_pressure = 7, t_min = 1200, t_max = 1600, p_min = 0.1, p_max = 10)    # 设置初始温度压强采样点
    md.set_time_step(delta_t = 1e-7)    # 设置cantera模拟的步长
    md.load_true_data()                 # 加载真实数据

    # 生成删10个0的向量数据, self.current_vector可以查看这些数据
    initial_zero_num = 10
    md.generate_vector(zero_num = initial_zero_num, size = 1000, generate_all = False)
    # md.load_ignition_data('/home/wangzhiwei/combustion_CH4/data/ignition_data/idt_1000data_10zero.npz')
    # 假设我们希望一直训练到删除210个反应
    max_zero_num = 150
    for i in range(100):
        if i == 0:
            md.load_ignition_data('/home/wangzhiwei/combustion_CH4/data/ignition_data/idt_1000data_10zero.npz')
        else:
            args = {'num_temperature': md.num_temperature, 'num_pressure': md.num_pressure,
                    'train_input': md.current_vector,
                    't_min': md.t_min, 't_max': md.t_max, 'p_min': md.p_min, 'p_max': md.p_max,
                    'zero_num': md.current_zero_num,
                    'process': 32,
                    'save_path': md.ignition_data_path,
                    'fuel': md.fuel, 'oxidizer': md.oxidizer,
                    'input_file': md.mechanism,
                    'save' : True}
            current_idt_data_path = generate_idt_data(args)
            md.load_ignition_data(current_idt_data_path)
        
        # 将点火数据转化为DNN训练数据
        md.generate_DNN_data(rate = 0.8)

        args = {'learning_rate': 5e-4, 'lr_decay_step': 100, 'lr_decay_rate': 0.7, 
                'batch_size': 128,
                'hidden_units': [600, 800, 1000],
                'epoch': [2000],
                'super_epoch': 10}
        # 设置dnn参数和训练dnn
        md.set_dnn_args(args)
        md.dnn_train(mod = 0, gpu_index = 2)
        md.set_dnn_args(args)
        md.dnn_train(mod = 1, gpu_index = 2)
        md.set_dnn_args(args)
        md.dnn_train(mod = 2, gpu_index = 2)
        # 生成向量数据 保存数据, 可以调用md.current_vector查看生成的01矩阵
        md.generate_vector(zero_num = initial_zero_num + 2 * i + 2, size = 100000, save = False)
        # dnn预测并返回合适的矩阵,可以调用md.current_vector查看更改后的01矩阵
        md.dnn_predict(tol = 0.01, target_size = 500, save = True)
    print('over!')



def full_procedure2():
    md = model_reduction('./chem.yaml')
    md.set_FO('H2','O2')
    md.sample_TP(num_temperature = 9, num_pressure = 7, t_min = 1200, t_max = 1600, p_min = 0.1, p_max = 10)
    md.set_time_step(delta_t = 1e-7)
    md.load_true_data()
    l = list(item for item in os.listdir('/home/wangzhiwei/combustion_dnn/reduction/data/idt_data'))
    for files in l:
        target_file = os.path.join('/home/wangzhiwei/combustion_dnn/reduction/data/idt_data', files)
        md.load_ignition_data(target_file)
    
        md.generate_DNN_data(rate = 0.8)
        args = {'learning_rate': 5e-4, 'lr_decay_step': 100, 'lr_decay_rate': 0.9, 
                'batch_size': 32,
                'hidden_units': [500, 500, 500],
                'epoch': [3200], 'super_epoch': 10}
        md.set_dnn_args(args)
        md.dnn_train(mod = 1, gpu_index = 3)
    print('over!')


def full_procedure3():
    md = model_reduction('./chem.yaml')
    md.set_FO('H2','O2')
    md.sample_TP(num_temperature = 9, num_pressure = 7, t_min = 1200, t_max = 1600, p_min = 0.1, p_max = 10)
    md.set_time_step(delta_t = 1e-7)
    md.GenTrueData()
    
    for i in range(20):
        size  = 100000 + 20000 * i
        md.generate_vector(zero_num = i + 3, size = size, save = False)
        md.dnn_predict(tol = 0.001, target_size = 3000, gpu_index = 3, save = True)
        args = {'num_temperature': md.num_temperature, 'num_pressure': md.num_pressure,
                'train_input': md.current_vector,
                't_min': md.t_min, 't_max': md.t_max, 'p_min': md.p_min, 'p_max': md.p_max,
                'zero_num': md.current_zero_num,
                'process': 32,
                'save_path': md.ignition_data_path,
                'fuel': md.fuel, 'oxidizer': md.oxidizer,
                'input_file': md.mechanism,
                'save' : True}
        generate_idt_data(args)
    print('over!')


def full_procedure4():
    md = model_reduction('./chem.yaml')
    md.set_FO('H2','O2')
    md.sample_TP(num_temperature = 9, num_pressure = 7, t_min = 1200, t_max = 1600, p_min = 0.1, p_max = 10)
    md.set_time_step(delta_t = 1e-7)
    l = list(item for item in os.listdir('/home/wangzhiwei/combustion_dnn/reduction/data/dnn_data'))
    for files in l:
        target_file = os.path.join('/home/wangzhiwei/combustion_dnn/reduction/data/dnn_data', files)
        md.current_dnn_data_path = target_file
        Data = np.load(md.current_dnn_data_path)
        md.x_train,     md.x_test     = Data['x_train'],     Data['x_test']
        md.y_train_idt, md.y_train_T, = Data['y_train_idt'], Data['y_train_T']
        md.y_test_idt,  md.y_test_T,  = Data['y_test_idt'],  Data['y_test_T']

        args = {'learning_rate': 5e-4, 'lr_decay_step': 100, 'lr_decay_rate': 0.9, 
                'batch_size': 32,
                'hidden_units': [500, 500, 500],
                'epoch': [3200], 'super_epoch': 10}
        md.set_dnn_args(args)
        md.dnn_train(mod = 0, gpu_index = 2)
    print('over!')



# 用class中的solve_inverse函数求解反问题，并且设计代码评估反问题质量
def inverse():
    md = model_reduction('/home/wangzhiwei/combustion/code/chem.yaml')
    md.set_FO('H2','O2')
    md.sample_TP(num_temperature = 9, num_pressure = 7, t_min = 1200, t_max = 1600, p_min = 0.1, p_max = 10)
    md.set_time_step(delta_t = 1e-7)
    # md.load_true_data()
    x_0 = 0.3 * np.random.rand(1,27) + 0.7
    # x_0 = np.ones((1, 27))

    # ------------------------
    x = md.solve_inverse(x_0, lr = 1e-3, t = 0.5, gpu_index = 1, iteration = 100000)
    # ------------------------

    # x = np.ones(27)
    # 画图评估反问题得到解的质量
    # 真实点火延迟时间
    true_Data = np.load('%s/true_idt_data.npz'% (md.true_data_path))
    true_idt = np.log10(np.array(true_Data['IDT']))
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
    IDT, Temperature = [], []
    initial_temperature = np.linspace(md.t_min, md.t_max, num = md.num_temperature, endpoint = True)
    initial_pressure = np.exp(np.linspace(np.log(md.p_min), np.log(md.p_max), num = md.num_pressure, endpoint = True))
    for i in range(md.num_temperature):
        for j in range(md.num_pressure):
            # 实例化gas对象，必须每次重新实例化
            gas = ct.Solution(thermo='ideal-gas', kinetics='gas', species=species, reactions=reactions)
            gas.TP = initial_temperature[i], initial_pressure[j] * ct.one_atm
            gas.set_equivalence_ratio(1, md.fuel, md.oxidizer)
            # 求点火延迟时间
            idt, T = solve_idt(gas, md.fuel, delta_t=1e-7)
            IDT.append(idt)
            Temperature.append(T)
            print('i=%s, j=%s, idt=%s' % (i, j ,idt))
    IDT = np.log10(IDT)
    print(IDT)
    print(true_idt)
    print('initial x:', x_0)
    x = [-7, -3]
    y = [-7, -3]
    plt.plot(x,y,'r--',lw = 1)
    plt.scatter(IDT, true_idt, c = 'b', s = 2)
    plt.ylabel('true_idt')
    plt.xlabel('reduced_idt')
    # plt.xscale('log')
    # plt.yscale('log')
    plt.savefig('./pic/inverse_iter10000_1.png')

def true_ignition_data():
    md = model_reduction('./gri.yaml')
    md.set_FO('CH4','O2')
    md.sample_TP(num_temperature = 9, num_pressure = 7, t_min = 1200, t_max = 1600, p_min = 0.1, p_max = 10)
    md.set_time_step(delta_t = 1e-7)
    md.GenTrueData()
    print('done!')







if __name__ == "__main__":

    full_procedure4()






