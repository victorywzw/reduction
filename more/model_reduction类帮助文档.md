# model_reduction Class

## 类简介

输入一个详细机理的yaml文件，将通过神经网络预测简化机理，且简化机理在0维燃烧的点火延迟和最终火焰温度均十分接近详细机理的计算结果。

## 类属性

#### self.mechanism

详细机理文件chem.yaml的路径

#### self.gas

```python
self.gas = ct.Solution(self.mechanism)
```

#### self.all_species

详细机理中涉及的所有组分

```python
self.all_species = ct.Species.listFromFile(self.mechanism)
```

#### self.all_reactions

详细机理中涉及的所有反应

```python
self.all_reactions = ct.Reaction.listFromFile(self.mechanism, self.gas)
```

#### self.species_num

详细机理的组分数

#### self.reactions_num

详细机理的反应数

#### self.delta_t

cantera计算0维点火采用的时间步长，默认1e-7。

#### self.fuel

燃料，如"H2"

#### self.oxidizer

氧化剂，如"O2"

#### self.initial_temperature

模拟的初始温度列表

#### self.initial_pressure

模拟的初始压强列表

#### self.current_vector

当前所考虑的所有简化机理对应的0-1向量。

#### self.current_zero_num

当前所考虑的所有简化机理对应的0-1向量的0的个数，即删除反应的数量。

#### self.current_vector_size

当前所考虑的所有简化机理的个数。

#### self.current_vector_path

当前生成的01向量的保存路径

#### self.all_idt_data

当前所考虑的所有简化机理计算的在T-P采样点上的点火延迟时间。（T-P采样点即不同初始温度和初始压强构成的采样点）

#### self.reduced_reactions

当前所考虑的一条简化机理涉及的反应。（内部计算涉及到的属性，几乎没有调用价值）

#### self.reduced_species

当前所考虑的一条简化机理涉及的组分。（内部计算涉及到的属性，几乎没有调用价值）

#### self.num_pressure

初始压强采样点的个数。

#### self.num_temperature

初始温度采样点的个数。

#### self.x_train

神经网络的输入（训练集）。

#### self.y_train

神经网络的真实输出（训练集）。

#### self.x_test

神经网络的输入（测试集）。

#### self.y_test

神经网络的真实输出（测试集）。

#### self.current_dnn_data_path

神经网络使用的数据集的路径。

#### self.data_path = './data'

数据路径。

#### self.dnn_data_path = './data/dnn_data'

用于神经网络训练和测试的数据路径。

#### self.idt_data_path = './data/idt_data'

cantera模拟的点火延迟数据路径。

#### self.loss_data_path = './data/loss_data'

神经网络训练的loss数据路径。

#### self.true_data_path = './data/true_data'

真实点火延迟数据的路径。

#### self.vector_data_path = './data/vector_data'

用于生成简化机理的0-1向量存放路径。

#### self.model_path = './model'

神经网络模型存放路径。

#### self.model_json_path = './model/model_json'

神经网络模型超参数存放路径。

#### self.model_pth_path = './model/model_pth'

神经网络预训练模型存放路径。

#### self.pic_path = './pic'

各种结果图片保存路径。



## 类函数

#### mechanism_info(self)

通过传入的详细机理yaml文件更新类中的属性。

#### print_info(self)

输出详细机理的组分，反应等信息。

#### set_FO(self, fuel, oxidizer)

设置燃料和氧化物

#### set_F(self, fuel)

设置燃料

#### set_O(self, oxidizer)

设置氧化物

#### set_time_step(self, delta_t = 1e-7)

设置cantera模拟的时间步长

#### set_dnn_args(self, args)

设置神经网络的超参数，运行后更新self.args。

#### load_dnn_data(self, dnn_data_path)

加载dnn的训练测试数据集，运行后更新self.current_dnn_data_path, self.x_train, self.y_train, self.x_test, self.y_test属性

#### sample_TP(self, num_temperature, num_pressure, t_min, t_max, p_min, p_max)

设置初始温度压强采样点

#### generate_vector(self, zero_num, size = 10, generate_all = False, save = True)

生成指定0个数的01向量，运行后更新self.current_vector, self.current_zero_num, self.current_vector_size

#### dnn_predict(self, gpu_index = 3, rate = 0.15, save = True)

输入01向量，使用预训练过的神经网络进行预测，挑选前rate * 100%个01向量。运行后更新self.current_vector, self.current_vector_size。

#### solve_idt(self, gas, delta_t = 1e-6)

用cantera计算gas的点火延迟时间和最终火焰温度，二者将被作为该函数的输出。

#### generate_IdtData(self)

生成self.current_vector对应的点火数据。运行后更新self.all_idt_data, self.current_vector, self.current_vector_size, self.current_data_path.

更新self.current_vector的原因是cantera在计算过程中有时会报错，导致整个向量报废，需要剔除。

#### GenOneData(self, tmp_path, index)

生成一个01向量的点火延迟数据。是self.generate_IdtData()的子函数。

#### GenTrueData(self)

生成真实的点火延迟和最终火焰温度数据，并将其保存在路径self.true_data_path之下。

#### generate_DNN_data(self, rate = 0.8)

将当前的点火数据转化成dnn数据，运行后更新self.current_dnn_data_path, self.x_train, self.y_train, self.x_test, self.y_test。rate是训练集占总数据集的比重。

#### dnn_train(self, gpu_index = 2)

训练神经网络，如果路径中存在预训练模型的setting.json文件将自动加载，否则将从头训练。调用的训练参数为self.x_train, self.y_train, self.x_test, self.y_test，以及超参数self.args。运行后将保存超参数至setting.json，且分别以pth和npz形式保存model和loss history。

### machanism2vector(self, sub_reactions)

将简化机理转化为01向量，若详细机理的反应在简化机理中出现过，则记为1，否则记为0。

#### vector2machanism(self, vector)

将01向量转化为简化机理，原理与machanism2vector相同。

#### generate_dir(self)

在代码所在的目录下生成data, model, pic三大目录，分别保存数据，网络模型和图像。

#### rm系列

删除指定文件夹下所有文件

#### evaluate_one_mechanism(self, vector)

评估一个机理的好坏

#### evaluate_mechanism(self, vector)

评估一组机理的好坏



## example

这里列举一些调用该class的例子

### example1:使用model_reduction做机理简化

本例给出了一个完整的使用model_reduction class完成机理简化的流程，首先以所有删除3个反应的简化机理训练dnn，之后不断删除更多反应，用预训练的dnn初步筛选这些简化机理，之后代入cantera进行计算，得到的数据集用于训练dnn，依此不断迭代，最终生成较好的dnn预测器和简化机理。

因为类与多进程存在一些矛盾和我水平的限制，暂时无法调用多进程生成简化机理的点火数据，因此删除3个反应的简化机理代入cantera计算耗时很大。后边因为有了dnn的初步筛选，因此耗时还可以接受。

```python
# 实例化，我们考虑的是一个27个反应的机理文件
# 此时在当前代码目录下会产生文件夹
md = model_reduction('/home/wangzhiwei/combustion/code/chem.yaml')

# 设置燃料和氧化剂
md.set_FO('H2','O2')

# 设置初始温度压强采样点
md.sample_TP(num_temperature = 9, num_pressure = 7, t_min = 1600, t_max = 1200, p_min = 0.1, p_max = 10)

md.set_time_step(delta_t = 1e-7)

# 生成真实数据（这个函数尚未完备，可以暂时将true_idt_data.npz文件放入true_data文件夹下来替代这个函数）
# md.generate_true_data()

# 生成删3个0的向量数据
# self.current_vector可以查看这些数据
md.generate_vector(zero_num = 3, generate_all = True)

# 这里18意味着01向量中0的个数最终会到18+3=21个，我们考虑的机理只有27个反应，因此已经很大了
for i in range(18):
      # 生成点火数据, 将npz保存到file里
      md.generate_IdtData()

      # 将点火数据转化为DNN训练数据
      md.generate_DNN_data(rate = 0.8)

      args={
        'learning_rate': 1e-4,
        'lr_decay_step': 50000,
        'lr_decay_rate': 0.95, 
        'hidden_units': [500, 500, 500],
        'epoch': [1000],
        'super_epoch': 500
      }

      # 调整/设置dnn参数
      md.set_dnn_args(args)

      md.dnn_train(gpu_index = 2)

      # 生成向量数据 保存数据 
      # 可以调用md.current_vector查看生成的01矩阵
      md.generate_vector(zero_num = i + 4, size = 100000, save = False)

      # dnn预测并返回合适的矩阵
      # 可以调用md.current_vector查看更改后的01矩阵
      md.dnn_predict(save = True)

  print('over!')
```

### example2:生成指定数据并喂给已经训练过的dnn继续训练

如果我们希望在已有的dnn基础上手动生成数据并喂给dnn，我们可以用以下代码完成。这相对于example1有更多可调空间。

```python
# 实例化class
md = model_reduction('/home/wangzhiwei/combustion/code/chem.yaml')
md.set_FO('H2','O2')
md.sample_TP(num_temperature = 9, num_pressure = 7, t_min = 1600, t_max = 1200, p_min = 0.1, p_max = 10)
md.set_time_step(delta_t = 1e-7)

# 生成、筛选数据（dnn预测）
md.generate_vector(zero_num = 5, size = 100, generate_all = False)
md.dnn_predict(save = True)
md.generate_IdtData(process_num = 8)
md.generate_DNN_data(rate = 0.8)

# 喂数据（dnn训练）
args={
    'learning_rate': 1e-4,
    'lr_decay_step': 50000,
    'lr_decay_rate': 0.95, 
    'hidden_units': [500, 500, 500],
    'epoch': [1000],
    'super_epoch': 500
}

# 调整/设置dnn参数
md.set_dnn_args(args)

md.dnn_train(gpu_index = 2)
```

### example3:将已有数据喂给dnn继续训练

如果我们已经有了一些数据，想喂给dnn训练，我们可以调用如下代码。这对于开始时无法使用多进程生成数据导致大量时间消耗具有重要改进。我们可以在类外写一个多进程程序来快速生成初始的数据，然后用md.load_dnn_data传入类中进行训练。

```python
md = model_reduction('/home/wangzhiwei/combustion/code/chem.yaml')
md.set_FO('H2','O2')
md.sample_TP(num_temperature = 9, num_pressure = 7, t_min = 1600, t_max = 1200, p_min = 0.1, p_max = 10)
md.set_time_step(delta_t = 1e-7)

md.load_dnn_data('/home/wangzhiwei/reduction/data/dnn_data/5zero_size16206_rate0.8.npz')

# 喂数据（dnn训练）
args={
    'learning_rate': 1e-3,
    'lr_decay_step': 50000,
    'lr_decay_rate': 0.95, 
    'hidden_units': [500, 500, 500],
    'epoch': [1000],
    'super_epoch': 500
}
# 调整/设置dnn参数
md.set_dnn_args(args)
md.dnn_train(gpu_index = 2)
```



## 尚未完成的改进

1. 点火延迟数据和火焰温度数据即使取了log可能还会存在量级上的差距，可能需要一个尺度因子修正。
