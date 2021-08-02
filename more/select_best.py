# -*- coding:utf-8 -*-
import numpy as np
import json
from MyNet import MyNet
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch

    
def dnn_predict(pre_input, DNN_FolderName, save_path, tol = 1,
                true_data_path = '/home/wangzhiwei/combustion/idt_data/true_idt/true_idt_data.npz'):
    '''
        神经网络预测输出，并选择误差小于tol的输出
        pre_input:输入的矩阵
        DNN_FolderName:DNN所在的位置，用于加载模型
        save_path:保存符合要求的矩阵
        tol:预测点火延迟与真实点火延迟差的模长
        true_data_path:真实点火数据位置
    '''

    print('loading true idt data......')
    # 真实点火延迟时间
    true_Data = np.load(true_data_path)
    true_idt = np.log10(np.array(true_Data['IDT']))

    print('loading dnn parameter......')
    # 加载配置文件
    with open('%s/model/settings.json' % (DNN_FolderName),'r',encoding='utf8')as fp:
        json_data = json.load(fp)
    fp.close()
    # 实例化DNN，加载已训练的网络参数
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    my_model = MyNet(json_data).to(device)
    checkpoint = torch.load('%s/model/my_model%s.pth' % (DNN_FolderName, json_data['train_index']))
    my_model.load_state_dict(checkpoint['model'])
    criterion = nn.MSELoss(reduction='mean')

    print('dnn predicting......')
    predict_idt = my_model(torch.FloatTensor(pre_input).to(device)).cpu().detach().numpy()

    print('select best......')
    print('tolerance = %s' % (tol))
    # 设置一个tolerance，保留效果小于tolerance的向量
    reduced_vector = []
    Err = []
    num = np.size(predict_idt, 0)
    for i in range(num):
        dnn_idt = predict_idt[i]
        error = np.linalg.norm(dnn_idt - true_idt)
        Err.append(error)
        if error < tol:
            reduced_vector.append(pre_input[i])
        if i % 10000 == 0:
            print(i)
    size = len(reduced_vector)
    print('input %s vectors, output %s vectors ' % (num, size))

    np.savez('%s/vector_%snum_%stol.npz' % (save_path, size, tol), 
            reduced_vector = reduced_vector, tol = tol)  


if __name__ == "__main__":

    DNN_FolderName = '/home/wangzhiwei/combustion/result/20210516_025412'
    save_path = '/home/wangzhiwei/combustion/vector'
    tol = 0.85

    num = 100000
    vector_len = 27

    # 随机生成至多zero_num个0的0-1向量
    zero_num = 7
    pre_input = np.ones((num, vector_len))
    for i in range(num):
        zero_index = np.random.randint(0, vector_len, zero_num)
        for j in zero_index:
            pre_input[i,j] = 0

    dnn_predict(pre_input, DNN_FolderName, save_path, tol)




