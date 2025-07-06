# FedGNN 
// /usr/lib/wsl/lib/libcuda.so
FedGNN是一个基于联邦学习的图神经网络(Graph Neural Network)推荐系统实现。该项目主要用于在保护用户隐私的情况下，进行协同过滤推荐任务。该系统通过联邦学习框架分布式训练用户-物品交互模型，同时采用加密技术保护用户数据隐私。

# 1. 快速开始
1. Environment Requirements
* Ubuntu 16.04
* Anaconda with Python 3.6.9 // Python 3.10.19 
* CUDA 10.0

Note: The specific python package list of our environment is included in the requirements.txt. The tensorflow version can be 1.12~1.15.
The installation may need several minutes if there is no environmental conflicts.

2. Hardware requirements
Needs a Linux server with larger than 32GB memory. GPU cores are optional but recommended.

3. Running

* Download datasets from their original sources
* Convert then into matlab matrix formats (rows: users, columns: items, value: ratings)
* Execute "python run.py"

Note: The logs will show the training loss and the test results. The estimated running time is from tens of minutes to hours, depending on the dataset size.


# 2. 各模块详细分析
## 2.1. 核心模型架构 (model.py)
该文件定义了神经网络模型结构：

- 使用Keras构建了一个基于GNN的推荐系统模型
- 包含用户嵌入层(userembedding_layer)和物品嵌入层(itemembedding_layer)
- 实现了图注意力机制(GAT)，通过attention_gat聚合邻居节点信息
- 使用了 ComputeMasking 自定义层处理填充值
- 模型采用梯度下降优化器(SGD)，使用均方误差(MSE)作为损失函数
- 模型支持隐私保护机制，通过梯度裁剪(clipnorm)限制模型更新
## 2.2. 数据预处理 (preprocess.py)
该模块负责数据准备工作：

- generate_history: 构建用户交互历史记录，并进行随机打乱
- generate_training_data: 从训练矩阵中提取用户-物品交互数据
- generate_test_data: 从测试矩阵中提取评估数据 
- 所有数据处理都考虑了评分标准化（通过LABEL_SCALE常量）
## 3. 图扩展模块 (expansion.py)
该模块处理图结构的构建与扩展：

- graph_embedding_expansion 函数实现了基于用户交互的图嵌入扩展
- 使用加密机制保护用户交互信息
- 构建了一个用户-物品-用户的二分图结构
- 通过第三方服务器协调用户之间的间接连接，但不泄露原始数据
- 为每个用户构建邻居嵌入矩阵，用于后续模型训练

## 4. 加密模块 (encrypt.py)
实现了数据加密保护机制：

- 使用RSA非对称加密算法保护用户数据
- 提供了密钥生成、签名验证和加解密功能
- generate_ke: 生成公钥和私钥 
- sign 和 verify: 用于数据签名和验证
-encrypt_data 和 decrypt_data: 用于数据加密和解密

## 5. 数据生成器 (generator.py)
负责批量数据生成：

- generate_batch_data_random: 为训练过程生成随机批次数据
- generate_batch_data: 为测试过程生成批次数据
- 这些生成器是Keras模型训练的数据提供者
## 6. 常量定义 (const.py)
包含全局常量参数：

- LABEL_SCALE = 100: 评分标准化比例
- HIDDEN = 64: 隐藏层维度
- DROP = 0.2: Dropout比例
- BATCH_SIZE = 32: 批次大小
- HIS_LEN = 100: 历史记录长度
- PSEUDO = 1000: 伪交互项目数
- NEIGHBOR_LEN = 100: 邻居长度
- CLIP = 0.2: 梯度裁剪参数
- LR = 0.01: 学习率
- EPS = 0.1: 差分隐私参数
- EPOCH = 10: 训练轮数
## 7. 工具函数 (utils.py)
包含辅助函数（从导入情况看主要用于数据加载和处理）。

## 8. 主程序 (run.py)
项目的主入口文件：

- 加载并预处理数据集（从MATLAB格式文件）
- 构建用户交互历史和图结构
- 生成加密密钥
- 构建和训练模型
- 执行测试评估，使用RMSE(均方根误差)作为评估指标
- 实现了差分隐私机制，通过拉普拉斯噪声保护模型参数

技术特点
- 联邦学习框架：通过分布式方式训练推荐模型，用户数据不需要共享
- 图神经网络：利用用户-物品交互图结构进行深度学习
- 隐私保护：
  - 使用RSA加密保护用户交互数据
  - 实现差分隐私机制，通过添加噪声保护模型参数
  - 使用第三方服务器进行安全计算
- 注意力机制：使用图注意力网络(GAT)聚合邻居节点信息

总体而言，FedGNN是一个结合了联邦学习、图神经网络和隐私保护技术的推荐系统实现，旨在在不共享原始数据的情况下提供高质量的推荐结果。

# 3. FedGNN执行流程详解
## 3.1整体流程概述
FedGNN是一个联邦学习框架下的图神经网络推荐系统，其执行流程可以概括为以下几个主要步骤：

- 数据准备：加载用户-物品交互矩阵
- 预处理：构建用户历史交互和图结构
- 加密配置：生成RSA公私钥对
- 模型构建：创建GNN推荐模型
- 训练流程：执行联邦学习训练
- 测试评估：评估模型性能

## 3.2可控制的变量

### 1. 数据集

通过修改 run.py 中的path_dataset变量，可以切换不同的数据集：
```python
# 以下是可选的数据集路径
path_dataset = ('/home/maxiaoming/project/FedPerGNN-main/mgcnn-master/Data/douban/training_test_dataset.mat')
path_dataset = ('/home/maxiaoming/project/FedPerGNN-main/mgcnn-master/Data/flixster/training_test_dataset_10_NNs.mat')
path_dataset = ('/home/maxiaoming/project/FedPerGNN-main/mgcnn-master/Data/movielens/split_1.mat')
path_dataset = ('/home/maxiaoming/project/FedPerGNN-main/mgcnn-master/Data/synthetic_netflix/synthetic_netflix.mat')
path_dataset = ('/home/maxiaoming/project/FedPerGNN-main/mgcnn-master/Data/yahoo_music/training_test_dataset_10_NNs.mat')
```

### 2. 超参数
在 const.py 中定义了多个可调整的超参数：

- LABEL_SCALE = 100：评分标准化比例
- HIDDEN = 64：隐藏层维度
- DROP = 0.2：Dropout比例
- BATCH_SIZE = 32：批次大小
- HIS_LEN = 100：历史记录长度
- PSEUDO = 1000：伪交互项目数
- NEIGHBOR_LEN = 100：邻居长度
- CLIP = 0.2：梯度裁剪参数
- LR = 0.01：学习率
- EPS = 0.1：差分隐私参数
- EPOCH = 10：训练轮数

## 3.3详细执行步骤
### 步骤1：环境准备
首先需要确保环境符合要求：

- Ubuntu 16.04
- Python 3.6.9 (Anaconda环境)
- CUDA 10.0
- 安装 requirements.txt 中的依赖包

```bash
# 在项目目录下执行
pip install -r requirements.txt
```
作用：确保所有依赖库和工具已正确安装，为后续执行提供环境保障。

### 步骤2：数据获取与准备
FedGNN使用MATLAB格式的矩阵文件作为输入：

- 获取原始数据集
- 将数据转换为MATLAB矩阵格式
- 确保数据格式为：行=用户，列=物品，值=评分

作用：提供算法所需的用户-物品交互数据，是整个推荐系统的基础。

### 步骤3：配置数据路径
修改
run.py
文件中的path_dataset变量，指向您的数据集位置：

```python
# 设置为您本地的数据集路径
path_dataset = ('您的数据集路径/dataset_name.mat')
```
作用：指定要使用的具体数据集，便于在不同数据集间切换进行实验。

### 步骤4：执行训练与测试
运行主程序开始训练和测试过程：

```bash
python run.py
```
当执行这个命令后，系统会按以下流程运行：

#### 4.1 加载数据（run.py中的load_matlab_file函数）
```python
M = load_matlab_file(path_dataset, 'M')
Otraining = load_matlab_file(path_dataset, 'Otraining')
Otest = load_matlab_file(path_dataset, 'Otest')
```
- M：完整的用户-物品评分矩阵
- Otraining：训练集交互矩阵
- Otest：测试集交互矩阵

作用：加载数据集并区分训练和测试数据。

#### 4.2 数据预处理（preprocess.py中的函数）
```python
usernei = generate_history(Otraining)
trainu, traini, trainlabel, train_user_index = generate_training_data(Otraining, M)
testu, testi, testlabel = generate_test_data(Otest, M)
```
作用：构建用户交互历史、准备训练和测试数据集。这个步骤将原始评分矩阵转换为模型可用的格式。

#### 4.3 生成加密密钥（encrypt.py中的generate_key函数）
```python
generate_key()
```
作用：生成RSA公私钥对，为后续的安全通信和数据保护提供基础。

#### 4.4 构建模型（model.py中的get_model函数）
```python
model, userembedding_layer, itemembedding_layer = get_model(Otraining)
```
作用：创建图神经网络模型，定义了整个推荐系统的架构。

#### 4.5 训练模型（run.py中的train函数）
```python
user_neighbor_emb = train(model)
```
训练过程中的关键步骤：

1. 获取当前用户嵌入
2. 通过 graph_embedding_expansion 扩展图结构
3. 生成批量训练数据
4. 进行模型批次训练
5. 添加噪声（实现差分隐私）
6. 周期性进行测试评估

作用：通过联邦学习方式更新模型参数，同时保护用户隐私。

#### 4.6 测试评估（run.py中的test函数）
```python
test(model, user_neighbor_emb)
```
作用：在测试集上评估模型性能，计算均方根误差(RMSE)。

## 执行流程中各组件的具体作用
1. 图扩展模块（ expansion.py ）
- 在联邦学习框架下，允许用户共享加密后的交互信息
- 通过第三方服务器构建更全面的用户-物品-用户连接图
- 生成邻居嵌入表示，丰富模型的输入信息
2. 加密模块（ encrypt.py ）
- 生成和管理RSA密钥对
- 对用户交互数据进行签名和验证
- 确保数据在传输和处理过程中的安全性
3. 数据生成器（ generator.py ）
- 为训练和测试过程提供批量数据
- 确保数据随机性和均衡性
- 优化训练效率和模型泛化能力
4. 模型架构（ model.py ） 
- 定义GNN推荐模型结构 
- 实现图注意力机制聚合邻居信息 
- 构建端到端的推荐预测流程
5. 训练过程（ run.py train 函数） 实
- 现联邦学习的参数更新逻辑 
- 添加拉普拉斯噪声实现差分隐私
- 控制训练过程和评估周期

运行结果解读 
执行完成后，系统会输出：

- 训练过程中的损失值变化
- 测试集上的RMSE评估结果
较低的RMSE值表示模型预测更准确，推荐质量更高。通常评分预测任务中，RMSE低于1.0被认为是较好的结果。

这个系统的创新点在于结合了联邦学习、图神经网络和隐私保护技术，在不共享原始数据的情况下，仍能提供高质量的推荐结果。

