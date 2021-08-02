import matplotlib
import matplotlib.pyplot as plt 
import numpy as np
import os
import load_data


def plot_loss_f(file_path):
    '''
        按顺序加载文件夹下所有loss的npz文件并画图
    '''
    train_loss_his = []
    test_loss_his = []
    epoch_index = []
    num = 0
    result_path = os.path.join(file_path, 'result')
    l = list(item for item in os.listdir(result_path))
    list.sort(l)
    for files in l:
            target_file = os.path.join(result_path, files)
            data = np.load(target_file)
            print('loading.. %s' % (target_file))
            epoch_index.extend(data['epoch_index'])
            train_loss_his.extend(data['train_his'])
            test_loss_his.extend(data['test_his'])
            num += 1
    # print(epoch_index)
    
    plt.plot(epoch_index, train_loss_his, lw=2, label='train')
    plt.plot(epoch_index, test_loss_his, 'r--', lw=2, label='test')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig('%s/pic/loss_his%s.png' % (file_path, num))

def plot_loss_l(file_list, save_path):
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



if __name__=="__main__":

    # file_path = '/home/wangzhiwei/combustion/result/20210516_025412'
    # plot_loss_f(file_path)

    file_list = [
        '/home/wangzhiwei/combustion/result/20210516_025412/result/loss_his2.npz',
        '/home/wangzhiwei/combustion/result/20210516_025412/result/loss_his3.npz',
        '/home/wangzhiwei/combustion/result/20210516_025412/result/loss_his4.npz'
    ]
    save_path = '/home/wangzhiwei/combustion/result/20210516_025412/pic/loss_23and4.png'

    plot_loss_l(file_list, save_path)
    





