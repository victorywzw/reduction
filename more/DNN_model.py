import numpy as np
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F

import func_tools


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





class DNN_model():
    def __init__(self):
    
    def set_dnn_args(self, args):
        pass

    def dnn_train(self,
            Hyperparameter,
            dnn_data_list,
            model_path = '/home/wangzhiwei/reduction/model',
            gpu_index = 2,
            use_current_lr = False
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

        settings = '%s/model_json/settings.json'
        # 判断第一次训练还是有预训练模型
        if os.path.exists(settings):
            # 加载配置文件
            with open(settings, 'r', encoding='utf8')as fp:
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
    
    def dnn_predict(
            self, 
            pre_input, 
            DNN_FolderName = '..', 
            save_path = '../data/dnn_data', 
            tol = 1,
            true_data_path = '../data/true_data/true_idt_data.npz'
        ):
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



