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

# 设置神经网络结构
# 只接受三层隐藏层的输入
class MyNet(nn.Module):
    # 构造方法里面定义可学习参数的层
    def __init__(self, input_dim, hidden_units, output_dim):
        super(MyNet, self).__init__()

        # 定义隐藏层
        self.fc1  = nn.Linear(input_dim, hidden_units[0])
        self.fc2  = nn.Linear(hidden_units[0], hidden_units[1])
        self.fc3  = nn.Linear(hidden_units[1], hidden_units[2])
        self.fc4  = nn.Linear(hidden_units[2], output_dim, bias = False)

        self.block = nn.Sequential()
        self.block.add_module('linear1', self.fc1)
        self.block.add_module('relu', nn.ReLU())
        self.block.add_module('linear2', self.fc2)
        self.block.add_module('relu', nn.ReLU())
        self.block.add_module('linear3', self.fc3)
        self.block.add_module('relu',nn.ReLU())
        self.block.add_module('linear4', self.fc4)

    # forward定义输入数据进行前向传播
    def forward(self, x):
        return self.block(x)


# 调用时应当把chem.yaml与代码放到同一文件夹下
class model_reduction():
    def __init__(self, mechanism = './chem.yaml',):
        self.mechanism = mechanism
        self.mechanism_info()
        self.generate_dir()
        self.set_FO('H2','O2')
        self.sample_TP(num_temperature = 9, num_pressure = 7, t_min = 1600, t_max = 1200, p_min = 0.1, p_max = 10)
        self.set_time_step(delta_t = 1e-7)
    '''============================================================================================================='''
    '''                                                初始化设置                                                    '''
    '''============================================================================================================='''
    # 获取机理信息
    def mechanism_info(self):
        self.gas = ct.Solution(self.mechanism)
        self.all_species   = ct.Species.listFromFile(self.mechanism)
        self.all_reactions = ct.Reaction.listFromFile(self.mechanism, self.gas)
        self.species_num   = len(self.all_species)      # 组分个数
        self.reactions_num = len(self.all_reactions)    # 反应个数

    # 输出机理信息
    def print_info(self):
        print('mechanism_path:', self.mechanism)
        print('Species: {0}'.format(', '.join(S.name for S in self.all_species)))
        print('Reactions:')
        for i, R in enumerate(self.all_reactions):
            print('%s. ' % i, R.equation)
    
    # 指定燃料和氧化剂
    def set_FO(self, fuel, oxidizer):
        self.fuel, self.oxidizer = fuel, oxidizer

    # 设置模拟时间步长
    def set_time_step(self, delta_t = 1e-7):
        self.delta_t = delta_t

    # 设置初始温度压强采样点
    def sample_TP(self, num_temperature, num_pressure, t_min, t_max, p_min, p_max):
        self.t_min, self.t_max, self.p_min, self.p_max = t_min, t_max, p_min, p_max
        self.num_pressure, self.num_temperature, self.sample_num = num_pressure, num_temperature, num_pressure * num_temperature
        # 温度等距采样，压强log后等距采样
        self.initial_temperature = np.linspace(t_min, t_max, num = num_temperature, endpoint = True)
        self.initial_pressure = np.exp(np.linspace(np.log(p_min), np.log(p_max), num = num_pressure, endpoint = True))

    '''============================================================================================================='''
    '''                                           true_data and vector_data                                         '''
    '''============================================================================================================='''
    # 给定预混气体gas，返回该气体的点火延迟时间。需要指定燃料fuel
    def solve_idt(self, gas, delta_t = 1e-6):
        r = ct.IdealGasReactor(gas)
        sim = ct.ReactorNet([r])
        time, max_hrr, idt = 0.0, 0.0, 0.0
        final_fuel_fraction = r.thermo[self.fuel].X[0] * 0.3
        now_fuel_fraction = r.thermo[self.fuel].X[0]
        now_temperature, previous_temperature = r.T, r.T
        # 模拟的终止条件是：温度不变或燃料消耗指定值或燃料不再消耗
        while abs(previous_temperature - now_temperature) > 1e-3 or now_fuel_fraction > final_fuel_fraction or abs(previoous_fuel_fraction - now_fuel_fraction) > 1e-4:

            previoous_fuel_fraction, previous_temperature = r.thermo[self.fuel].X[0], r.T
            time += delta_t
            sim.advance(time)
            now_fuel_fraction, now_temperature = r.thermo[self.fuel].X[0], r.T
            
            # 以反应过程中，最大的温度变化率作为点火延迟时间点
            dTdt = (now_temperature - previous_temperature)/delta_t
            if dTdt > max_hrr:
                max_hrr, idt = dTdt, time
            if time > 0.1:
                if idt < 1e-8:
                    idt = 0.1
                break
        return idt, r.T, max_hrr

    # 生成真实点火数据
    def GenTrueData(self):
        IDT, Temperature, HRR = [], [], []
        t0 = time.time()
        for i in range(self.num_temperature):
            for j in range(self.num_pressure):
                # 实例化gas对象，必须每次重新实例化
                self.gas.TP = self.initial_temperature[i], self.initial_pressure[j] * ct.one_atm
                self.gas.set_equivalence_ratio(1, self.fuel, self.oxidizer)

                # 求点火延迟时间
                idt, T, hrr = self.solve_idt(self.gas, self.delta_t)
                IDT.append(idt)
                Temperature.append(T)
                HRR.append(hrr)
        self.true_idt_data, self.true_T_data, self.true_hrr_data = IDT, Temperature, HRR
        print('生成真实数据耗时:{} s'.format(time.time() - t0))
        np.savez('%s/true_idt_data.npz' % (self.true_data_path), IDT = IDT, T = Temperature, HRR = HRR)
    
    # 加载真实点火数据
    def load_true_data(self):
        Data = np.load('%s/true_idt_data.npz' % (self.true_data_path))
        self.true_idt_data, self.true_T_data, self.true_hrr_data = Data['IDT'], Data['T'], Data['HRR']

    # 生成指定0数量的01向量
    def generate_vector(self, zero_num, size = 10, generate_all = False, save = True):
        print('generate_vector')
        if generate_all and zero_num == 3: # 生成所有符合要求的01向量
            size = 1
            for i in range(zero_num):
                size *= (self.reactions_num - i) / (i+1)
            array = np.ones([int(size), self.reactions_num])
            t = 0
            for i in range(self.reactions_num -2):
                for j in range(i+1, self.reactions_num - 1):
                    for k in range(j+1, self.reactions_num):
                        array[t, i], array[t, j], array[t, k] = 0, 0, 0
                        t += 1
        else:
            array = np.ones([size, self.reactions_num])
            for i in range(size):
                index = np.random.choice(self.reactions_num, zero_num, replace = False)
                for j in index:
                    array[i,j] = 0
        # 将生成的0-1矩阵传给self
        self.current_vector = array.tolist()
        self.current_zero_num = zero_num
        self.current_vector_size = len(self.current_vector)
        if save == True:
            self.currnet_vector_path = '%s/%szero_%ssize' % (self.vector_data_path, self.current_zero_num, self.current_vector_size)
            np.savez(self.currnet_vector_path, vector = self.current_vector, zero_num = self.current_zero_num)
        print('done!')
    
    # 加载已经存在的ignition data
    def load_ignition_data(self, ignition_data_path):
        self.current_ignition_data_path = ignition_data_path
        Data = np.load(self.current_ignition_data_path)
        self.all_idt_data     = Data['all_idt_data']
        self.all_T_data       = Data['all_T_data']
        self.all_hrr_data     = Data['all_hrr_data']
        self.current_vector   = Data['all_vector']
        self.current_zero_num = Data['zero_num']
        self.current_vector_size = len(self.current_vector)
    '''============================================================================================================='''
    '''                                           dnn train and predict                                             '''
    '''============================================================================================================='''
    # 设置dnn超参数
    def set_dnn_args(self, args):
        self.args = args.copy()

    # 加载已经存在的dnn data 
    def load_dnn_data(self, dnn_data_path):
        self.current_dnn_data_path = dnn_data_path
        Data = np.load(self.current_dnn_data_path)
        self.x_train,     self.x_test                      = Data['x_train'],     Data['x_test']
        self.y_train_idt, self.y_train_T, self.y_train_hrr = Data['y_train_idt'], Data['y_train_T'], Data['y_train_hrr']
        self.y_test_idt,  self.y_test_T,  self.y_test_hrr  = Data['y_test_idt'],  Data['y_test_T'],  Data['y_test_hrr']

    # dnn预测
    def dnn_predict(self, gpu_index = 2, batch_size = 50000, target_size = 5000, tol = 0, save = True, output = False):
        '''
            神经网络预测输出，并选择误差小于tol的输出
            output = True : 输出预测值
            target_size: dnn_idt与true_idt最接近的前target_size个将被选出
            tol:dnn_idt < tol的简化机理也将被选出
        '''
        pre_input = np.array(self.current_vector) # 网络输入
        origin_length = self.current_vector_size
        # 加载点火延迟的网络
        if os.path.exists('%s/settings.json' % (self.model_idt_json_path)):
            # 加载配置文件与实例化DNN
            with open('%s/settings.json' % (self.model_idt_json_path),'r',encoding='utf8')as fp:
                json_data = json.load(fp)
            fp.close()
            device = torch.device("cuda:%s" % (gpu_index) if torch.cuda.is_available() else "cpu")
            my_model = MyNet(json_data['input_dim'], json_data['hidden_units'], json_data['output_dim']).to(device)
            checkpoint = torch.load('%s/my_model%s.pth' % (self.model_idt_pth_path, json_data['train_index']), map_location = device)
            my_model.load_state_dict(checkpoint['model'])
            
            print('dnn predicting...')     # 批量预测，防止内存不足
            t0 = time.time()
            batch_num = self.current_vector_size // batch_size
            for i in range(batch_num):
                start, end = i * batch_size, (i + 1) * batch_size
                if i == 0:
                    predict_idt = my_model(torch.FloatTensor(pre_input[start : end,:]).to(device)).cpu().detach().numpy()
                else:
                    tmp = my_model(torch.FloatTensor(pre_input[start : end,:]).to(device)).cpu().detach().numpy()
                    predict_idt = np.vstack((predict_idt, tmp))
            if self.current_vector_size % batch_size != 0:
                start = batch_num * batch_size
                tmp = my_model(torch.FloatTensor(pre_input[start:,:]).to(device)).cpu().detach().numpy()
                predict_idt = np.vstack((predict_idt, tmp))

            # 为提高泛化能力，我们首先筛选一些能够点起火的数据，这些数据的idt不必与true_idt接近
            reduced_vector = []
            count = 0
            for i in range(self.current_vector_size):
                fail = 0
                for j in range(self.sample_num):
                    if predict_idt[i, j] > tol:  
                        fail += 1
                if fail < 10: # 如果在小于10个采样点上都能点的起火，则采用它
                    reduced_vector.append(self.current_vector[i])
                    count += 1
                if count > target_size - 1:
                    break


            # 接下来选择dnn_idt与true_idt相近的前target_size个向量
            # 真实点火延迟时间
            true_Data = np.load('%s/true_idt_data.npz'% (self.true_data_path))
            true_idt = np.log10(np.array(true_Data['IDT']))
            Err_idt = []
            for i in range(self.current_vector_size):  # 得到真实idt与dnn预测的idt的误差列表
                error = np.linalg.norm(predict_idt[i] - true_idt)
                Err_idt.append(error)
            predict_idt = predict_idt.tolist()

            # 对列表按Err进行排序
            Num = list(range(0, self.current_vector_size))
            zip_data = list(zip(Err_idt, Num, predict_idt))
            zip_data.sort()
            Err_idt[:], Num[:], predict_idt[:] = zip(*zip_data)

            # 筛选误差最小的前target_size个01向量
            reduced_vector_2 = []
            for i in range(target_size):
                num = Num[i]
                reduced_vector_2.append(self.current_vector[num])

            self.current_vector = list(reduced_vector + reduced_vector_2)
            self.current_vector_size = len(self.current_vector)
            time_cost = time.time() - t0
            print('%s original vectors input, %s + %s vectors are selected' % (origin_length, len(reduced_vector), len(reduced_vector_2)))
            print('time cost:{:4e}s'.format(time_cost))

        if save == True:
            np.savez('%s/pre_%szero_%ssize.npz' % (self.vector_data_path, self.current_zero_num, self.current_vector_size), 
                    vector      = self.current_vector, 
                    vector_size = self.current_vector_size,
                    zero_num    = self.current_zero_num)  
        
        if output == True:
            return predict_idt[:self.current_vector_size]

    # 能够将当前数据快速转化成DNN的数据集
    def generate_DNN_data(self, rate = 0.8):
        
        # 将得到的所有的数据按同一顺序打乱
        zip_data = list(zip(self.current_vector, self.all_idt_data, self.all_T_data, self.all_hrr_data))
        random.shuffle(zip_data)
        self.current_vector[:], self.all_idt_data[:], self.all_T_data, self.all_hrr_data = zip(*zip_data)

        # 将数据列表转化为矩阵
        current_vector = np.array(self.current_vector)
        all_idt_data   = np.array(self.all_idt_data)
        all_T_data     = np.array(self.all_T_data)
        all_hrr_data   = np.array(self.all_hrr_data)
        
        train_size = int(rate * self.current_vector_size)
        # 训练集
        self.x_train     = current_vector[0: train_size, :]
        self.y_train_idt = np.log10(all_idt_data[0: train_size, :])
        self.y_train_T   = np.log10(all_T_data[0: train_size, :])
        self.y_train_hrr = np.log10(all_hrr_data[0: train_size, :] + 0.01) # 0.01防止log10无意义

        # 测试集
        self.x_test     = current_vector[train_size: , :]
        self.y_test_idt = np.log10(all_idt_data[train_size: , :])
        self.y_test_T   = np.log10(all_T_data[train_size: , :])
        self.y_test_hrr = np.log10(all_hrr_data[train_size: , :] + 0.01)

        np.savez('%s/%ssize_%srate_%szero.npz' % (self.dnn_data_path, self.current_vector_size, rate, self.current_zero_num),
                x_train     = self.x_train,
                y_train_idt = self.y_train_idt,
                y_train_T   = self.y_train_T,
                y_train_hrr = self.y_train_hrr,
                x_test      = self.x_test,
                y_test_idt  = self.y_test_idt,
                y_test_hrr  = self.y_test_hrr)

        self.current_dnn_data_path = '%s/%ssize_%srate_%szero.npz' % (self.dnn_data_path, self.current_vector_size, rate, self.current_zero_num)
    

    # 训练DNN
    def dnn_train(self, mod = 0, gpu_index = 2):
        '''
            训练神经网络，参数说明：
                mod 0:训练预测idt的网络（log10尺度）; 1:训练预测T的网络, 2:训练预测hrr的网络
                gpu_index:使用哪个gpu跑程序，暂时只能设置一个
        '''
        # 数据
        input_dim  = np.size(self.x_train, 1)
        output_dim = np.size(self.y_train_idt, 1)
        train_size = np.size(self.x_train, 0)
        test_size  = np.size(self.x_test,  0)

        self.args['input_dim']     = input_dim
        self.args['output_dim']    = output_dim
        self.args['train_size']    = [train_size]
        self.args['test_size']     = [test_size]

        device = torch.device("cuda:%s" % (gpu_index) if torch.cuda.is_available() else "cpu")

        x_train, x_test = self.x_train, self.x_test

        # 判断训练idt的网络还是T的网络
        if mod == 0:    # 点火延迟网络
            model_json_path = self.model_idt_json_path
            model_pth_path  = self.model_idt_pth_path
            loss_data_path  = self.model_idt_loss_path
            y_train = self.y_train_idt
            y_test  = self.y_test_idt
        if mod == 1:    # 最终火焰温度网络
            model_json_path = self.model_T_json_path
            model_pth_path  = self.model_T_pth_path
            loss_data_path  = self.model_T_loss_path
            y_train = self.y_train_T
            y_test  = self.y_test_T
        if mod == 2:    # 热释放率网络
            model_json_path = self.model_hrr_json_path
            model_pth_path  = self.model_hrr_pth_path
            loss_data_path  = self.model_hrr_loss_path
            y_train = self.y_train_hrr
            y_test  = self.y_test_hrr


        settings = '%s/settings.json' % (model_json_path)
        # 判断第一次训练还是有预训练模型
        if os.path.exists(settings):
            # 加载配置文件
            with open(settings, 'r', encoding='utf8')as fp:
                json_data = json.load(fp)
            fp.close()
            self.args['lr_scheduler'] = json_data['lr_scheduler'] + [[self.args['learning_rate'], self.args['lr_decay_step'], self.args['lr_decay_rate']]]
            self.args['data_path']    = json_data['data_path'] + [self.current_dnn_data_path]
            self.args['train_index']  = json_data['train_index'] + 1
            self.args['epoch']        = json_data['epoch'] + self.args['epoch']
            self.args['train_size']   = json_data['train_size'] + self.args['train_size']
            self.args['test_size']    = json_data['test_size'] + self.args['test_size']
            self.args['batch_size_history'] = json_data['batch_size_history'] + [self.args['batch_size']]
            start_epoch = np.sum(json_data['epoch'])
            end_epoch   = np.sum(self.args['epoch'])
            # 实例化DNN，加载已训练的网络参数
            my_model   = MyNet(input_dim, self.args['hidden_units'], output_dim).to(device)
            checkpoint = torch.load('%s/my_model%s.pth' % (model_pth_path, self.args['train_index']-1), map_location = device)
            my_model.load_state_dict(checkpoint['model'])
        else: # 没有预训练网络的情形
            self.args['lr_scheduler']       = [[self.args['learning_rate'], self.args['lr_decay_step'], self.args['lr_decay_rate']]]
            self.args['data_path']          = [self.current_dnn_data_path]
            self.args['batch_size_history'] = [self.args['batch_size']]
            self.args['train_index']        = 1
            start_epoch, end_epoch = 0, self.args['epoch'][0]
            my_model = MyNet(input_dim, self.args['hidden_units'], output_dim).to(device)

        # 定义损失和优化器
        criterion = nn.MSELoss(reduction='mean')
        optimizer = optim.Adam(my_model.parameters(), lr=self.args['learning_rate'])
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=self.args['lr_decay_step'], gamma=self.args['lr_decay_rate'])

        ''' 训练神经网络 '''
        t0 = time.time()
        epoch_index, train_his, test_his = [], [], []
        self.args['batch_num'] =  train_size // self.args['batch_size']

        for epoch in range(start_epoch, end_epoch):
            my_model.train()
            # 按照 batch 进行训练
            train_loss = 0
            for i in range(self.args['batch_num']):
                x_train_batch = x_train[i * self.args['batch_size']: (i+1) * self.args['batch_size'] , : ]
                y_train_batch = y_train[i * self.args['batch_size']: (i+1) * self.args['batch_size'] , : ]

                y_dnn_train_batch = my_model(torch.FloatTensor(x_train_batch).to(device))
                train_loss_batch  = criterion(y_dnn_train_batch, torch.FloatTensor(y_train_batch).to(device))

                # 计算梯度与反向传播
                optimizer.zero_grad()
                train_loss_batch.backward()
                optimizer.step()
                train_loss += float(train_loss_batch)
            scheduler.step()

            train_loss /= self.args['batch_num'] 

            # 在特殊epoch处输出训练信息
            if epoch % self.args['super_epoch'] == 0:
                epoch_index.append(epoch)
                
                my_model.eval()
                with torch.no_grad():
                    y_dnn_test = my_model(torch.FloatTensor(x_test).to(device))
                    test_loss = float(criterion(y_dnn_test, torch.FloatTensor(y_test).to(device)))
                
                train_his.append(train_loss)
                test_his.append(test_loss)

                # 保存模型
                state = {'model':my_model.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch':epoch}
                torch.save(state, '%s/my_model%s.pth' % (model_pth_path, self.args['train_index']))

                print('epoch: {}\t train loss: {:.4e}   test loss: {:.4e}   time cost: {} s   lr:{:.2e}'
                    .format(epoch, train_loss, test_loss, int(time.time()-t0), optimizer.param_groups[0]['lr']))

        ''' 保存实验结果 '''
        # 现在的lr
        # self.args['learning_rate'] = optimizer.param_groups[0]['lr']

        # 删除一些不需要的参数
        self.args.pop('learning_rate')
        self.args.pop('lr_decay_step')
        self.args.pop('lr_decay_rate')

        # 保存DNN模型
        state = {'model':my_model.state_dict(), 
                'optimizer':optimizer.state_dict(), 
                'epoch':epoch}
        torch.save(state, '%s/my_model%s.pth' % (model_pth_path, self.args['train_index']))

        # 保存DNN的loss
        np.savez('%s/loss_his%s.npz' % (loss_data_path, self.args['train_index']), 
                epoch_index = epoch_index, train_his = train_his, test_his = test_his)

        # 保存DNN的超参数
        with open('%s/settings.json' % (model_json_path), "w") as f:
            f.write(json.dumps(self.args, ensure_ascii=False, indent=4, separators=(',', ':')))
        f.close()
    


    
    def plot_loss(self, file_path):
        '''
            按顺序加载文件夹下所有loss的npz文件并画图
        '''
        train_loss_his = []
        test_loss_his = []
        epoch_index = []
        num = 0
        l = list(item for item in os.listdir(file_path))
        list.sort(l)
        for files in l:
            target_file = os.path.join(file_path, files)
            data = np.load(target_file)
            print('loading.. %s' % (target_file))
            epoch_index.extend(data['epoch_index'])
            train_loss_his.extend(data['train_his'])
            test_loss_his.extend(data['test_his'])
            num += 1
    
        plt.plot(epoch_index, train_loss_his, lw=2, label='train')
        plt.plot(epoch_index, test_loss_his, 'r--', lw=2, label='test')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.xscale('log')
        plt.yscale('log')
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.savefig('%s/loss_his%s.png' % (self.pic_path, num))

    def plot_loss_l(self, file_list, save_path):
        '''
            传入npz文件路径组成的列表，画出loss
            用save_path手动指定保存路径和名字
        '''
        train_loss_his = []
        test_loss_his = []
        epoch_index = []
        
        for files in file_list:
            data = np.load(files)
            print('loading.. %s' % (files))
            epoch_index.extend(data['epoch_index'])
            train_loss_his.extend(data['train_his'])
            test_loss_his.extend(data['test_his'])

        plt.plot(epoch_index, train_loss_his, lw=2, label='train')
        plt.plot(epoch_index, test_loss_his, 'r--', lw=2, label='test')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.xscale('log')
        plt.yscale('log')
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.savefig(save_path)

    '''============================================================================================================='''
    '''                                              inverse problem                                                '''
    '''============================================================================================================='''
    def solve_inverse(self, x_0, lr = 1, t1 = 1, t2 = 1, t3 = 1, gpu_index = 0, 
                        iteration = 100, lr_decay_step = 100, lr_decay_rate = 0.99):

        device = torch.device("cuda:%s" % (gpu_index) if torch.cuda.is_available() else "cpu")

        # 真实点火延迟时间
        true_Data = np.load('%s/true_idt_data.npz'% (self.true_data_path))
        true_idt  = torch.FloatTensor(np.log10(np.array(true_Data['IDT']))).to(device)
        true_hrr  = torch.FloatTensor(np.log10(np.array(true_Data['HRR']) + 0.01)).to(device)
        true_T    = torch.FloatTensor(np.log10(np.array(true_Data['T']))).to(device)

        # 加载三个网络
        # 加载点火延迟的网络
        with open('%s/settings.json' % (self.model_idt_json_path),'r',encoding='utf8')as fp:
            json_data = json.load(fp)
        fp.close()
        model_idt = MyNet(json_data['input_dim'], json_data['hidden_units'], json_data['output_dim']).to(device)
        checkpoint = torch.load('%s/my_model%s.pth' % (self.model_idt_pth_path, json_data['train_index']), map_location = device)
        model_idt.load_state_dict(checkpoint['model'])
        # 加载热释放率的网络
        with open('%s/settings.json' % (self.model_hrr_json_path),'r',encoding='utf8')as fp:
            json_data = json.load(fp)
        fp.close()
        model_hrr = MyNet(json_data['input_dim'], json_data['hidden_units'], json_data['output_dim']).to(device)
        checkpoint = torch.load('%s/my_model%s.pth' % (self.model_hrr_pth_path, json_data['train_index']), map_location = device)
        model_hrr.load_state_dict(checkpoint['model'])
        # 加载最终温度的网络
        with open('%s/settings.json' % (self.model_T_json_path),'r',encoding='utf8')as fp:
            json_data = json.load(fp)
        fp.close()
        model_T = MyNet(json_data['input_dim'], json_data['hidden_units'], json_data['output_dim']).to(device)
        checkpoint = torch.load('%s/my_model%s.pth' % (self.model_T_pth_path, json_data['train_index']), map_location = device)
        model_T.load_state_dict(checkpoint['model'])

        criterion1 = nn.MSELoss(reduction='mean')
        criterion2 = nn.L1Loss()

        # 优化对象
        x_0 = x_0.squeeze()
        x = torch.autograd.Variable(torch.FloatTensor(x_0), requires_grad=True)
        optimizer = optim.SGD([x,], lr = lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = lr_decay_step, gamma = lr_decay_rate)

        zero_vector = torch.FloatTensor(np.zeros(self.reactions_num)).to(device)

        # 优化迭代
        for i in range(iteration):
            if i == 10000:
                print('predict_idt:\n', predict_idt, '\n true_idt', true_idt)
                print('predict_hrr:\n', predict_hrr, '\n true_hrr', true_hrr)
                print('predict_T:\n',   predict_T,   '\n true_T',   true_T)
            # 用神经网络预测
            x_01 = nn.Sigmoid(100 * x)
            predict_idt = model_idt(x_01.to(device))
            predict_hrr = model_hrr(x_01.to(device))
            predict_T   = model_T(x_01.to(device))
            loss1 = criterion1(predict_idt, true_idt)
            loss2 = criterion1(predict_hrr, true_hrr)
            loss3 = criterion1(predict_T, true_T)
            loss4 = criterion2(x_01.to(device), zero_vector)
            loss  = t1 * loss1 + t2 * loss2 + t3 * loss3 + loss4
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            if i % 1000 == 0:
                print('i = {}, loss1 = {:4e}, loss2 = {:4e}, loss3 = {:4e}, loss4 = {:4e}, loss = {:4e}, lr = {:2e}' 
                        .format(i, float(t1 * loss1), float(t2 * loss2), float(t3 * loss3), float(loss4), float(loss), optimizer.param_groups[0]['lr']))
        x = x.cpu().detach().numpy()

        print('predict_idt:\n', predict_idt, '\n true_idt', true_idt)
        print('predict_hrr:\n', predict_hrr, '\n true_hrr', true_hrr)
        print('predict_T:\n',   predict_T,   '\n true_T',   true_T)
        predict_idt = predict_idt.cpu().detach().numpy()
        predict_hrr = predict_hrr.cpu().detach().numpy()
        predict_T   = predict_T.cpu().detach().numpy()

        print('final x:\n',list(x))
        # 返回0-1向量
        onehot_x = 1 * (nn.Sigmoid(100 * x) > 0.8)
        print('final onehot_x:\n', list(onehot_x))

        # 保存训练结果
        args = {
            'x' : x, 
            'onehot_x'    : onehot_x, 
            'predict_idt' : predict_idt,
            'predict_hrr' : predict_hrr,
            'predict_T'   : predict_T
        }
        if not os.path.exists('./data/inverse_result'):
            os.mkdir('./data/inverse_result')

        if os.path.exists('./data/inverse_result/inverse_result.json'):
            with open('./data/inverse_result/inverse_result.json', 'r', encoding='utf8')as fp:
                json_data = json.load(fp)
            fp.close()
            json_data['index'] += 1
            key = 'experiment index: %s' % (json_data['index'])
            new_data = {key: args}
            json_data.update(new_data)
        else:
            json_data = {}
            json_data['index'] = 0
            key = 'experiment index: %s' % (json_data['index'])
            new_data = {key: args}
            json_data.update(new_data)

        with open('./data/inverse_result/inverse_result.json', "w", encoding="utf-8") as f:
            f.write(json.dumps(json_data, ensure_ascii=False, indent=4, separators=(',', ':')))
        f.close()
            
        return onehot_x, predict_idt, predict_T, predict_hrr

    def solve_inverse2(self, x_0, lr = 1, t1 = 1, t2 = 1, t3 = 1, gpu_index = 0, 
                    iteration = 100, lr_decay_step = 100, lr_decay_rate = 0.99):

        device = torch.device("cuda:%s" % (gpu_index) if torch.cuda.is_available() else "cpu")

        # 真实点火延迟时间
        true_Data = np.load('%s/true_idt_data.npz'% (self.true_data_path))
        true_idt  = torch.FloatTensor(np.log10(np.array(true_Data['IDT']))).to(device)
        true_T    = torch.FloatTensor(np.log10(np.array(true_Data['T']))).to(device)

        # 加载点火延迟的网络
        with open('%s/settings.json' % (self.model_idt_json_path),'r',encoding='utf8')as fp:
            json_data = json.load(fp)
        fp.close()
        model_idt = MyNet(json_data['input_dim'], json_data['hidden_units'], json_data['output_dim']).to(device)
        checkpoint = torch.load('%s/my_model%s.pth' % (self.model_idt_pth_path, json_data['train_index']), map_location = device)
        model_idt.load_state_dict(checkpoint['model'])
        # 加载最终温度的网络
        with open('%s/settings.json' % (self.model_T_json_path),'r',encoding='utf8')as fp:
            json_data = json.load(fp)
        fp.close()
        model_T = MyNet(json_data['input_dim'], json_data['hidden_units'], json_data['output_dim']).to(device)
        checkpoint = torch.load('%s/my_model%s.pth' % (self.model_T_pth_path, json_data['train_index']), map_location = device)
        model_T.load_state_dict(checkpoint['model'])

        criterion1 = nn.MSELoss(reduction='mean')
        criterion2 = nn.L1Loss()

        # 优化对象
        x_0 = x_0.squeeze()
        initial_x = x_0
        x = torch.autograd.Variable(torch.FloatTensor(x_0), requires_grad=True)
        optimizer = optim.SGD([x,], lr = lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = lr_decay_step, gamma = lr_decay_rate)

        zero_vector = torch.FloatTensor(np.zeros(self.reactions_num)).to(device)

        # 优化迭代
        Loss = []
        t = 0.001
        for i in range(iteration):
            f = nn.Sigmoid()
            x_01 = f(100 * x)
            if i == 0:
                print('\033[0;33;40minitial x:\033[0m ', x)
            predict_idt = model_idt(x_01.to(device))
            predict_T   = model_T(x_01.to(device))
            loss1 = criterion1(predict_idt, true_idt)
            loss2 = criterion1(predict_T, true_T)
            loss3 = criterion2(x_01.to(device), zero_vector)
            loss  = t1 * loss1 + t2 * loss2 + t3 * loss3
            Loss.append(float(loss))
            optimizer.zero_grad()
            loss.backward()
            # if i > 200:
            #     tmp = x.grad.cpu().detach().numpy()
            #     if np.linalg.norm(np.abs(tmp)) != 0:
            #         optimizer.param_groups[0]['lr'] = t * 1 / np.linalg.norm(np.abs(tmp))
            #     else:
            #         print('-' * 30)
            #         print(x)
            #         print('-' * 30)
            #         break
            if i % 1000 == 0:
                optimizer.param_groups[0]['lr'] *= 10
            if i < 200:
                optimizer.param_groups[0]['lr'] = 1e-3

            optimizer.step()
            scheduler.step()
            if i % 200 == 0:
                print('\033[0;33;40mx:\033[0m', x)
                print('\033[0;33;40mx.grad:\033[0m', x.grad)
                print('\033[0;33;40mi:{}\t loss1:{:.2e}\t loss2:{:.2e}\t loss3:{:.2e}\t loss:{:.2e}\t lr:{:.2e}\033[0m' 
                        .format(i, float(loss1), float(loss2), float(loss3), float(loss), optimizer.param_groups[0]['lr']))
                t = t * 0.95

        x = x.cpu().detach().numpy()

        print('predict_idt:\n', predict_idt, '\n true_idt', true_idt)
        print('predict_T:\n',   predict_T,   '\n true_T',   true_T)
        predict_idt = predict_idt.cpu().detach().numpy()
        predict_T   = predict_T.cpu().detach().numpy()

        print('final x:\n',list(x))
        # 返回0-1向量
        onehot_x = 1 * (1 / (1 + np.exp(-100 * x)) > 0.9)
        print('final onehot_x:\n', list(onehot_x))

        # 保存训练结果
        args = {
            'initial x'   : initial_x.tolist(),
            'x' : x.tolist(), 
            'onehot_x'    : onehot_x.tolist(), 
            'predict_idt' : predict_idt.tolist(),
            'predict_T'   : predict_T.tolist(),
        }
        if not os.path.exists('./data/inverse_result'):
            os.mkdir('./data/inverse_result')

        if os.path.exists('./data/inverse_result/inverse_result.json'):
            with open('./data/inverse_result/inverse_result.json', 'r', encoding='utf8')as fp:
                json_data = json.load(fp)
                fp.close()
                json_data['index'] += 1
                key = 'experiment index: %s' % (json_data['index'])
                new_data = {key: args}
                json_data.update(new_data)
        else:
            json_data = {}
            json_data['index'] = 0
            key = 'experiment index: %s' % (json_data['index'])
            new_data = {key: args}
            json_data.update(new_data)

        with open('./data/inverse_result/inverse_result.json', "w", encoding="utf-8") as json_f:
                json_f.write(json.dumps(json_data, ensure_ascii=False, indent=4, separators=(',', ':')))
        json_f.close()

        return onehot_x, predict_idt, predict_T, Loss



    # 输入子反应，输出对应的向量
    def machanism2vector(self, sub_reactions):
        vector = np.zeros(len(self.reactions_num))
        for i, R in enumerate(self.all_reactions):
            if R in sub_reactions:
                vector[i] = 1
        return vector

    # 输入向量，输出对应的子反应
    def vector2machanism(self, vector):
        sub_reactions = []
        for i, R in enumerate(self.all_reactions):
            if vector[i] == 0:
                continue
            sub_reactions.append(R)
        return sub_reactions


    '''============================================================================================================='''
    '''                                                 make file                                                   '''
    '''============================================================================================================='''
    def generate_dir(self):
        # 公共参数文件夹
        if not os.path.exists('./data'):
            os.mkdir('./data')
            os.mkdir('./data/dnn_data')
            os.mkdir('./data/ignition_data')
            os.mkdir('./data/true_data')
            os.mkdir('./data/vector_data')
            
        # 生成存放三种网络数据的文件夹
        if not os.path.exists('./model'):
            os.mkdir('./model')
        for file_name in ['idt', 'T', 'hrr']:
            if not os.path.exists('./model/model_%s' % file_name):  
                os.mkdir('./model/model_%s' % file_name)
                os.mkdir('./model/model_%s/model_json' % file_name)
                os.mkdir('./model/model_%s/model_pth' % file_name)
                os.mkdir('./model/model_%s/loss_his' % file_name)

        if not os.path.exists('./pic'):
            os.mkdir('./pic')
        
        self.data_path          = './data'
        self.dnn_data_path      = './data/dnn_data'
        self.ignition_data_path = './data/ignition_data'
        self.true_data_path     = './data/true_data'
        self.vector_data_path   = './data/vector_data'

        self.model_idt_path      = './model/model_idt'
        self.model_idt_json_path = './model/model_idt/model_json'
        self.model_idt_pth_path  = './model/model_idt/model_pth'
        self.model_idt_loss_path = './model/model_idt/loss_his'
        self.model_T_path        = './model/model_T'
        self.model_T_json_path   = './model/model_T/model_json'
        self.model_T_pth_path    = './model/model_T/model_pth'
        self.model_T_loss_path   = './model/model_T/loss_his'
        self.model_hrr_path      = './model/model_hrr'
        self.model_hrr_json_path = './model/model_hrr/model_json'
        self.model_hrr_pth_path  = './model/model_hrr/model_pth'
        self.model_hrr_loss_path = './model/model_hrr/loss_his'
        self.pic_path = './pic'





if __name__ == "__main__":

    pass