import matplotlib
import matplotlib.pyplot as plt 
import os
import numpy as np
import json
from MyNet import MyNet
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
from load_data import DNN_Data


def distribution_01(Data, DNN_FolderName, save_path, plot_err = True, 
                    plot_distribution = False, num = 10):
    '''
        输入一个dnn的数据集和一个DNN，查看使得DNN的test_loss大的向量对应的01分布
        最终结果将通过直方图呈现
            save_path决定输出图像的保存位置
            plot_err控制是否输出排序后的err，一般先选为ture看一下大loss的比重，以确定num
            plot_distribution控制是否输出大loss对应的0-1向量的总体分布
            num决定前多少个被视为大loss
    '''
    print('loading data......')
    x_test = Data.x_test
    y_test = Data.y_test

    vector_num = np.size(x_test, 0)
    vector_len = np.size(x_test, 1)


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
    
    y_dnn_test = my_model(torch.FloatTensor(x_test).to(device)).cpu().detach().numpy()

    err = np.sum(np.abs(y_dnn_test - y_test) ** 2, axis = 1)
    err = list(err)
    vector_list = list(x_test)

    
    def takeFirst(elem):
        return elem[0]
    
    # 将得到的所有的数据按err从大到小顺序排列
    zip_data = list(zip(err, vector_list))
    zip_data.sort(key=takeFirst, reverse = True)
    err, vector_list = zip(*zip_data)


    if plot_err:
        plt.plot(err)
        plt.savefig('%s/err.png' % (save_path))
    
    if plot_distribution:
        # 计算分布
        vector = np.array(vector_list)
        distribution = list(np.sum(vector[:num,:], axis = 0))
        x = list(np.linspace(1, vector_len, vector_len))
        
        fig = plt.figure()
        pic1 = fig.add_subplot(111)
        plt.plot(x, distribution)
        for i,j in enumerate(x):
            plt.text(x[i], distribution[i], int(x[i]))
        plt.xticks(x)
        

        plt.savefig('%s/distribution.png' % (save_path))
    print('done!')
        


    


    



if __name__ == "__main__":

    dnn_data = DNN_Data()
    dnn_data.extend_data('/home/wangzhiwei/combustion/idt_data/dnn_data/dnndata_size8555_rate0.8.npz')

    DNN_FolderName = '/home/wangzhiwei/combustion/result/20210516_025412'

    save_path = '/home/wangzhiwei/combustion/code'

    distribution_01(dnn_data, DNN_FolderName, save_path, plot_err = True, 
                    plot_distribution = True, num = 4)















