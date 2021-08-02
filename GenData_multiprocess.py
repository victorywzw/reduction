# -*- coding:utf-8 -*-

import os
import shutil
import numpy as np
import cantera as ct
import time
import random
import multiprocessing as mp
from multiprocessing import Pool

from model_reduction import model_reduction

'''
    给定一个根机理文件的所有反应，以及一个子机理文件的所有反应，输出子机理对应的0-1向量，形式为numpy一维数组  
'''
def machanism2vector(base_reactions, sub_reactions):
    vector = np.zeros(len(base_reactions))
    for i, R in enumerate(base_reactions):
        if R in sub_reactions:
            vector[i] = 1
    return vector

'''
    给定一个根机理文件的所有反应，以及一个0-1向量(形式为numpy一维数组)，输出0-1向量对应的子机理的reaction和spacies
'''
def vector2machanism(base_reactions, vector):
    sub_reactions = []
    for i, R in enumerate(base_reactions):
        if vector[i] == 0:
            continue
        sub_reactions.append(R)
    return sub_reactions


# 返回点火延迟和最终火焰温度数据
def solve_idt(gas, fuel, delta_t = 1e-6):
    r = ct.IdealGasReactor(gas)
    sim = ct.ReactorNet([r])
    time, max_hrr, idt = 0.0, 0.01, 0.0
    final_fuel_fraction, now_fuel_fraction = r.thermo[fuel].X[0] * 0.3, r.thermo[fuel].X[0]
    now_temperature, previous_temperature  = r.T, r.T
    # 模拟的终止条件是：温度不变或燃料消耗指定值或燃料不再消耗
    while abs(previous_temperature - now_temperature) > 1e-3 or now_fuel_fraction > final_fuel_fraction or abs(previoous_fuel_fraction - now_fuel_fraction) > 1e-4:
        previoous_fuel_fraction, previous_temperature = r.thermo[fuel].X[0], r.T
        time += delta_t
        sim.advance(time)
        now_fuel_fraction, now_temperature = r.thermo[fuel].X[0], r.T
        # 以反应过程中，最大的温度变化率作为点火延迟时间点
        dTdt = (now_temperature - previous_temperature)/delta_t
        if dTdt > max_hrr:
            max_hrr, idt = dTdt, time
        if time > 0.05:
            if idt < 1e-8:
                idt = 0.1
            break
    return idt, r.T , max_hrr 


'''
    使用多进程生成0维点火延迟数据
'''
# 生成一个点火延迟数据
def GenOneData(tmp_path, index, train_input, input_file, fuel, oxidizer, num_temperature, num_pressure, 
                initial_temperature, initial_pressure, delta_t = 1e-7):
    # 加载详细机理文件
    all_species   = ct.Species.listFromFile(input_file)
    ref_phase     = ct.Solution(thermo='ideal-gas', kinetics='gas', species=all_species)
    all_reactions = ct.Reaction.listFromFile(input_file, ref_phase)
    
    tmp_vector = train_input[index]
    reactions = vector2machanism(all_reactions, tmp_vector)  # 获得简化机理涉及的反应
    species_names = {}  # 获得简化机理涉及的组分
    for reaction in reactions:
        species_names.update(reaction.reactants)
        species_names.update(reaction.products)
    species = [ref_phase.species(name) for name in species_names]
    # 生成简化机理在目标状态点上的点火延迟时间和最终火焰温度
    try:
        t0 = time.time()
        IDT, Temperature, HRR = [], [], []
        for i in range(num_temperature):
            for j in range(num_pressure):
                # 实例化gas对象，必须每次重新实例化
                gas = ct.Solution(thermo='ideal-gas', kinetics='gas', species=species, reactions=reactions)
                gas.TP = initial_temperature[i], initial_pressure[j] * ct.one_atm
                gas.set_equivalence_ratio(1, fuel, oxidizer)
                idt, T, hrr = solve_idt(gas, fuel, delta_t = delta_t) # 计算
                IDT.append(idt)
                Temperature.append(T)
                HRR.append(hrr)
        time_cost = time.time() - t0
    except:
        print('something wrong, but don\'t worry, I can not  handle it!')
    else:
        print('cost %s s for generate data of mechanism %s' % (time_cost, index))
        np.savez('%s/%sth.npz' % (tmp_path, index), IDT=IDT, T=Temperature, HRR=HRR, tmp_vector=tmp_vector)


def generate_idt_data(args):
    
    train_input = args['train_input']  # 简化机理01向量
    process     = args['process']
    save_path   = args['save_path']

    if 'delta_t' in args:
        delta_t = args['delta_t']
    else:
        delta_t = 1e-7

    input_file = args['input_file']
    fuel, oxidizer = args['fuel'], args['oxidizer']
    
    # 生成温度、压力
    t_min, t_max, p_min, p_max    = args['t_min'], args['t_max'], args['p_min'], args['p_max']
    num_temperature, num_pressure = args['num_temperature'], args['num_pressure']
    initial_temperature = np.linspace(t_min, t_max, num = num_temperature, endpoint = True)
    initial_pressure    = np.exp(np.linspace(np.log(p_min), np.log(p_max), num = num_pressure, endpoint = True))

    # 生成保存临时数据的文件夹
    tmp_path = os.path.join(save_path, 'datacache')
    if not os.path.exists(tmp_path):
        os.mkdir(tmp_path)
    
    # 多进程计算
    p = Pool(process)
    input_size = np.size(train_input, 0)
    for index in range(input_size):
        p.apply_async(func=GenOneData, args=(tmp_path, index, train_input, input_file, fuel, oxidizer, 
                    num_temperature, num_pressure, initial_temperature, initial_pressure, delta_t,))
    p.close()
    p.join()
    print('over')

    # 汇总并删除临时数据
    all_idt_data, all_T_data, all_hrr_data, all_vector = [], [], [], []  # 对每个简化机理求解采样点上的idt

    for files in os.listdir(tmp_path):
        target_file = os.path.join(tmp_path, files)
        if target_file.find('.npz'):
            tmp = np.load(target_file)    
            all_idt_data.append(tmp['IDT'])
            all_T_data.append(tmp['T'])
            all_hrr_data.append(tmp['HRR'])
            all_vector.append(tmp['tmp_vector'])
    shutil.rmtree(tmp_path, ignore_errors=True) # 删除中间文件

    if ('save' in args) and (args['save'] == True):
        zero_num = args['zero_num']
        data_num = len(all_vector)
        idt_data_path = '%s/idt_%sdata_%szero.npz' % (save_path, data_num, zero_num)
        np.savez(idt_data_path,
                all_idt_data = all_idt_data,
                all_T_data   = all_T_data,
                all_hrr_data = all_hrr_data,
                all_vector   = all_vector,
                zero_num     = zero_num)
        return idt_data_path
    else:
        return all_idt_data, all_T_data, all_hrr_data, all_vector

'''
    计算对冲火焰的稳态解。
'''
def solve_flame(gas):
    sim = ct.CounterflowPremixedFlame(gas = gas, width=0.2)

    sim.reactants.mdot = 0.12 # kg/m2/s
    sim.products.mdot  = 0.06 # kg/m2/s

    sim.set_refine_criteria(ratio=3, slope=0.1, curve=0.2)
    sim.solve(0, auto=True)
    return sim







if __name__ == "__main__":

    # 通过class中的函数生成01向量
    md = model_reduction('./chem.yaml')
    md.generate_vector(zero_num = 3, size = 10, generate_all = True, save = True)

    train_input = md.current_vector
    zero_num = md.current_zero_num

    # 参数字典
    args = {'num_temperature': 9,
            'num_pressure': 7,
            'train_input': train_input,
            't_min': 1200,
            't_max': 1600,
            'p_min': 0.1,
            'p_max': 10,
            'zero_num': zero_num,
            'process': 32,
            'save_path': './data/idt_data',
            'fuel': 'H2',
            'oxidizer': 'O2',
            'input_file': './chem.yaml'
            }
    

    t1 = time.time()
    generate_idt_data(args)
    t2 = time.time()
    print('generate chem cost time: {}s'.format(t2-t1))
