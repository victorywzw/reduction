import numpy as np
import sys, os
import random


class DNN_Data():

    def __init__(self):
        self.first = 0


    def gather_data(self, file_path = ['/home/wangzhiwei/combustion/idt_data/idt_63_4zero'], 
                    rate = 0.8, save_path = '/home/wangzhiwei/combustion/idt_data/dnn_data'):
        '''
            加载file_path列表中所有path对应文件夹下的npz文件，并将其同一保存在一个npz文件中
            file_path列表可以包含多个文件夹
            rate表示有多少用于当作训练集，有多少用于当作测试集
            该函数先load所有data，再把前 rate * 100% 个数据当作训练集，其他当作测试集
            return的字典Data中包含：
                train_input
                train_true_output
                test_input
                test_true_output
            类型均为矩阵
        '''
        Data = {}
        Data['input']  = []
        Data['output'] = []

        # 文件夹个数
        file_num = len(file_path)
        # 加载所有文件夹下的所有npz文件
        for file_i in range(file_num):
            for files in os.listdir(file_path[file_i]):
                target_file = os.path.join(file_path[file_i], files)
                if target_file.find('.npz'):
                    data = np.load(target_file)
                    print('loading.. %s' % (target_file))

                    Data['input'].extend(data['all_vector'])      # 网络输入，简化机理向量，epoch_num 行 vector_len 列
                    Data['output'].extend(data['all_state'])      # 真实数据，点火延迟时间，epoch_num 行 output_dim 列


        # 将得到的所有的数据按同一顺序打乱
        zip_data = list(zip(Data['input'], Data['output']))
        random.shuffle(zip_data)
        Data['input'][:], Data['output'][:] = zip(*zip_data)

        # 将数据列表转化为矩阵
        Data['input'] = np.array(Data['input'])
        Data['output'] = np.array(Data['output'])

        # 数据点个数
        data_size = np.size(Data['input'], 0)

        # 训练集
        train_size = int(rate * data_size)
        Data['train_input'] = Data['input'][0: train_size, :]
        Data['train_true_output'] = np.log10(Data['output'][0: train_size, :])

        # 测试集
        Data['test_input'] = Data['input'][train_size: , :]
        Data['test_true_output'] = np.log10(Data['output'][train_size: , :])

        np.savez('%s/dnndata_size%s_rate%s.npz' % (save_path, data_size, rate),
                train_input = Data['train_input'],
                train_true_output = Data['train_true_output'],
                test_input = Data['test_input'],
                test_true_output = Data['test_true_output'])
        print('save path: %s/dnndata_size%s_rate%s.npz' % (save_path, data_size, rate))



    def extend_data(self, npz_file):
        '''
            扩充训练集和测试集
            在原有训练集和测试集基础上，载入新的npz_file文件以扩充数据集
            如果之前没有数据集，那么将生成一个数据集
        '''
        Data = np.load(npz_file)

        if self.first == 0:
            self.x_train = Data['train_input']
            self.y_train = Data['train_true_output']
            self.x_test  = Data['test_input']
            self.y_test  = Data['test_true_output']
            self.first = 1
        else:
            self.x_train = np.vstack((self.x_train,Data['train_input']))
            self.y_train = np.vstack((self.y_train,Data['train_true_output']))
            self.x_test  = np.vstack((self.x_test,Data['test_input']))
            self.y_test  = np.vstack((self.y_test,Data['test_true_output']))
        print('extended!')
        


if __name__ == "__main__":
    
    file_path = [
        '/home/wangzhiwei/combustion/idt_data/idt_63_7zero'
    ]

    data = DNN_Data()
    data.gather_data(file_path)


    dnn_predict(pre_input, DNN_FolderName, save_path, tol)

    generate_idt_data(args)

    
















