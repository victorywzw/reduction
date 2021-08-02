'''
    评估器
    输入一个机理文件，
    输出点火延迟时间、火焰速度
'''

import numpy as np 
import cantera as ct
import random


import chemfile_transform



'''
    给定预混气体gas，返回该气体的点火延迟时间。
    需要指定燃料fuel
'''
def solve_idt(gas, fuel, delta_t = 1e-6):
    r = ct.IdealGasReactor(gas)
    sim = ct.ReactorNet([r])
    # delta_t = 1e-5
    time = 0.0
    max_hrr = 0.0
    idt = 0.0
    final_fuel_fraction = r.thermo[fuel].X[0] * 0.3
    now_fuel_fraction = r.thermo[fuel].X[0]
    now_temperature = r.T
    previous_temperature = r.T

    # 模拟的终止条件是：燃料耗尽且温度不再增加
    while abs(previous_temperature - now_temperature) > 1e-3 or now_fuel_fraction > final_fuel_fraction or abs(previoous_fuel_fraction - now_fuel_fraction) > 1e-4:
        
        previous_temperature = r.T
        previoous_fuel_fraction = r.thermo[fuel].X[0]
        
        time += delta_t
        sim.advance(time)
        
        now_fuel_fraction = r.thermo[fuel].X[0]
        now_temperature = r.T
        
        # 以反应过程中，最大的温度变化率作为点火延迟时间点
        dTdt = (now_temperature - previous_temperature)/delta_t
        if dTdt > max_hrr:
            max_hrr = dTdt
            idt = time
        if time > 0.1:
            if idt < 1e-8:
                idt = 0.1
            break
    return idt