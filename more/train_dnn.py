'''
    算法流程：
        1.事先采样好一个很大的数据库，(T,P,phi,idt,flame)
        2.神经网络输入简化机理，输出在这些(T,P,phi)上的表现(idt,flame)
        3.跟真实的(idt,flame)做比较(loss)
'''
# 自己的包
import func_tools
import chemfile_transform
import evaluation
from load_data import DNN_Data
from MyNet import MyNet

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


def main_train(
    Hyperparameter,
    dnn_data_list,
    FolderName = '/home/wangzhiwei/combustion/result/20210516_025412',
    gpu_index = 2,
    is_serious = 1,
    use_current_lr = False,
    first_train = 0
):
    '''
        训练神经网络，参数说明：
            Hyperparameter:超参数字典
            dnn_data_list:数据集路径列表
            FolderName:之前的实验结果的保存路径，如'/home/wangzhiwei/combustion/result/20210516_025412'
            gpu_index:使用哪个gpu跑程序，暂时只能设置一个
            is_serious:正式训练时为1，非正式训练(调试)改为0
            use_current_lr:False表示继续训练时指定新的学习率
            first_train:是否第一次训练
    '''

    '''
        加 载 数 据
    '''
    # 实例化对象
    dnn_data = DNN_Data()
    for data_path in dnn_data_list:
        dnn_data.extend_data(data_path)

    x_train = dnn_data.x_train
    y_train = dnn_data.y_train
    x_test  = dnn_data.x_test
    y_test  = dnn_data.y_test

    input_dim  = np.size(x_train, 1)
    output_dim = np.size(y_train, 1)
    train_size = np.size(x_train, 0)
    test_size  = np.size(x_test,  0)

    R['input_dim']     = input_dim
    R['output_dim']    = output_dim
    R['train_size']    = [train_size]
    R['test_size']     = [train_size]


    device = torch.device("cuda:%s" % (gpu_index) if torch.cuda.is_available() else "cpu")


    # 判断第一次训练还是有预训练模型
    if first_train == False:
        # 加载配置文件
        with open('%s/model/settings.json' % (FolderName), 'r', encoding='utf8')as fp:
            json_data = json.load(fp)
        fp.close()
        if use_current_lr == True:
            R['learning_rate'] = json_data['learning_rate']

        R['train_index'] = json_data['train_index'] + 1
        R['epoch'] = json_data['epoch'] + R['epoch']
        R['train_size'] = json_data['train_size'] + R['train_size']
        R['test_size']  = json_data['test_size'] + R['test_size']

        start_epoch = np.sum(json_data['epoch'])
        end_epoch   = np.sum(R['epoch'])
        # 实例化DNN，加载已训练的网络参数
        my_model = MyNet(R).to(device)
        checkpoint = torch.load('%s/model/my_model%s.pth' % (FolderName, R['train_index']-1), map_location = device)
        my_model.load_state_dict(checkpoint['model'])
    else:
        R['train_index'] = 1
        start_epoch = 0
        end_epoch = R['epoch'][0]

        # 创建目录
        FolderName = func_tools.mkdir_x('/home/wangzhiwei/combustion', is_serious = is_serious)
        # 实例化DNN
        my_model = MyNet(R).to(device)


    # 定义损失和优化器
    criterion = nn.MSELoss(reduction='mean')
    optimizer = optim.Adam(my_model.parameters(), lr=R['learning_rate'])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=R['lr_decay_step'], gamma=R['lr_decay_rate'])


    '''
        训练神经网络
    '''

    t0 = time.time()
    epoch_index, train_his, test_his = [], [], []

    for epoch in range(start_epoch, end_epoch):

        my_model.train()
        y_dnn_train = my_model(torch.FloatTensor(x_train).to(device))
        train_loss  = criterion(y_dnn_train, torch.FloatTensor(y_train).to(device))
        
        # 计算梯度与反向传播
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        scheduler.step()

        if epoch % R['super_epoch'] == 0:
            epoch_index.append(epoch)
            
            my_model.eval()
            with torch.no_grad():
                my_model.eval()
                y_dnn_test = my_model(torch.FloatTensor(x_test).to(device))
                test_loss  = float(criterion(y_dnn_test, torch.FloatTensor(y_test).to(device)))
            train_loss = float(train_loss)

            train_his.append(train_loss)
            test_his.append(test_loss)

            # 保存模型
            state = {'model':my_model.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch':epoch}
            torch.save(state, '%s/model/my_model%s.pth' % (FolderName, R['train_index']))

            print('epoch: {}\t train loss: {:.4e}   test loss: {:.4e}   time cost: {} s   lr:{:.2e}'
                .format(epoch, train_loss, test_loss, int(time.time()-t0), optimizer.param_groups[0]['lr']))


    '''
        保存实验结果，拷贝代码和修改文件名
    '''

    R['learning_rate'] = optimizer.param_groups[0]['lr']

    # 保存DNN模型
    state = {'model':my_model.state_dict(), 
            'optimizer':optimizer.state_dict(), 
            'epoch':epoch}
    torch.save(state, '%s/model/my_model%s.pth' % (FolderName, R['train_index']))


    # 保存DNN的超参数
    with open('%s/model/settings.json' % (FolderName), "w") as f:
        f.write(json.dumps(R, ensure_ascii=False, indent=4, separators=(',', ':')))
    f.close()


    # 保存DNN的loss
    np.savez('%s/result/loss_his%s.npz' % (FolderName, R['train_index']), 
            epoch_index = epoch_index, train_his = train_his, test_his = test_his)


    # 拷贝代码
    func_tools.savecode('.', '%s/code' % (FolderName))

    if first_train == 1:
        return FolderName


if __name__ == "__main__":
    
    '''
        设 置 超 参 数
    '''
    R={}

    R['learning_rate'] = 1e-4
    R['lr_decay_step'] = 50000
    R['lr_decay_rate'] = 0.95 
    R['hidden_units']  = [500, 500, 500]
    R['epoch']         = [1000000]
    R['super_epoch']   = 500

    # 训练DNN
    main_train(
        Hyperparameter = R,
        dnn_data_list = [
            '/home/wangzhiwei/combustion/idt_data/dnn_data/dnndata_size17756_rate0.8.npz'
        ],
        FolderName = None,
        gpu_index = 2,
        is_serious = 1,
        use_current_lr = False,
        first_train = 0
    )

