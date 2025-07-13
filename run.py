#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from utils import *
from encrypt import *
from model import *
from preprocess import *
from expansion import *
from generator import *
from const import *
import numpy as np
import random
import os
import datetime

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 创建models文件夹（如果不存在）
if not os.path.exists('./models'):
    os.makedirs('./models')
    print("创建models文件夹用于保存模型")

# 为当前训练会话创建时间戳文件夹
timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")
model_save_dir = os.path.join('./models', timestamp)
os.makedirs(model_save_dir, exist_ok=True)
print(f"本次训练结果将保存到: {model_save_dir}")

# 各种数据集路径选项
# path_dataset = ('./mgcnn-master/Data/douban/training_test_dataset.mat')
# path_dataset = ('./mgcnn-master/Data/flixster/training_test_dataset_10_NNs.mat')
# path_dataset = ('./mgcnn-master/Data/movielens/split_1.mat')
path_dataset = ('./mgcnn-master/Data/synthetic_netflix/synthetic_netflix.mat')


# path_dataset = ('./mgcnn-master/Data/yahoo_music/training_test_dataset_10_NNs.mat')

# In[ ]:


# 模型训练函数
def train(model):
    # 创建一个字典用于记录训练历史
    history = {'batchloss': [], 'epoch_rmse': []}
    
    # 遍历每个训练轮次
    for rounds in range(EPOCH):
        print(f"\n开始训练第 {rounds+1}/{EPOCH} 轮")
        # 获取当前用户嵌入矩阵
        alluserembs = userembedding_layer.get_weights()[0]
        # 执行图嵌入扩展（FedGNN的核心操作）
        user_neighbor_emb = graph_embedding_expansion(Otraining, usernei, alluserembs)
        # 生成随机批次训练数据
        traingen = generate_batch_data_random(BATCH_SIZE, train_user_index, trainu, traini, usernei, trainlabel,
                                              user_neighbor_emb)
        cnt = 0
        batchloss = []
        # 批次训练循环
        for i in traingen:
            # 获取当前模型权重
            layer_weights = model.get_weights()
            # 使用当前批次数据训练模型
            loss = model.train_on_batch(i[0], i[1])
            batchloss.append(loss)
            # 获取更新后的模型权重
            now_weights = model.get_weights()

            # 计算权重更新的标准差（用于差分隐私）
            sigma = np.std(now_weights[0] - layer_weights[0])
            # 添加伪交互项的噪声（差分隐私保护）
            norm = np.random.normal(0, sigma / np.sqrt(PSEUDO * BATCH_SIZE / now_weights[0].shape[0]),
                                    size=now_weights[0].shape)
            now_weights[0] += norm
            itemembedding_layer.set_weights([now_weights[0]])
            print("【当前批次平均损失】:", np.mean(batchloss))
            # 添加本地差分隐私噪声
            for i in range(len(now_weights)):
                now_weights[i] += np.random.laplace(0, LR * 2 * CLIP / np.sqrt(BATCH_SIZE) / EPS,
                                                    size=now_weights[i].shape)
            model.set_weights(now_weights)
            cnt += 1
            # 每10个批次打印一次进度
            if cnt % 10 == 0:
                print("【训练进度】批次:", cnt, "【当前损失】:", loss)
            # 达到一轮的批次上限后停止
            if cnt == len(train_user_index) // BATCH_SIZE:
                break
        # 每轮结束后测试模型性能
        rmse = test(model, user_neighbor_emb)
        history['epoch_rmse'].append(rmse)
        print("【本轮训练结束】【测试RMSE】:", rmse)
    
    # 训练结束后保存最终模型和训练历史
    save_model(model, userembedding_layer, itemembedding_layer, user_neighbor_emb, 
               os.path.join(model_save_dir, "fedgnn_final"))
    
    # 保存训练历史
    np.save(os.path.join(model_save_dir, 'training_history.npy'), history)
    print(f"训练历史保存至 {os.path.join(model_save_dir, 'training_history.npy')}")
    
    return user_neighbor_emb


# 模型测试函数
def test(model, user_neighbor_emb):
    # 生成测试数据批次
    testgen = generate_batch_data(BATCH_SIZE, testu, testi, usernei, testlabel, user_neighbor_emb)
    # 旧版TensorFlow用predict_generator，新版用predict
    # cr = model.predict_generator(testgen, steps=len(testlabel)//BATCH_SIZE+1,verbose=1)
    cr = model.predict(testgen, steps=len(testlabel) // BATCH_SIZE + 1, verbose=1)
    # 计算并打印RMSE评估指标（均方根误差）
    rmse = np.sqrt(np.mean(np.square(cr.flatten() - testlabel / LABEL_SCALE))) * LABEL_SCALE
    print('rmse:', rmse)
    return rmse


# 模型保存函数
def save_model(model, userembedding_layer, itemembedding_layer, user_neighbor_emb, save_path):
    """
    保存模型和嵌入向量
    Args:
        model: 训练好的FedGNN模型
        userembedding_layer: 用户嵌入层
        itemembedding_layer: 物品嵌入层
        user_neighbor_emb: 用户邻居嵌入
        save_path: 保存路径前缀
    """
    # 确保目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 尝试保存整个模型（如果支持）
    try:
        model.save(f"{save_path}_full.h5")
        print(f"完整模型已保存到: {save_path}_full.h5")
    except Exception as e:
        print(f"保存完整模型失败，改为保存架构和权重。错误: {e}")
        # 保存模型架构
        model_json = model.to_json()
        with open(f"{save_path}_model.json", "w") as json_file:
            json_file.write(model_json)
        # 保存权重
        model.save_weights(f"{save_path}_weights.h5")
        print(f"模型架构和权重已分别保存")
    
    # 保存训练配置信息
    with open(f"{save_path}_config.txt", "w") as f:
        f.write(f"训练时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"BATCH_SIZE: {BATCH_SIZE}\n")
        f.write(f"HIDDEN: {HIDDEN}\n")
        f.write(f"EPOCH: {EPOCH}\n")
        f.write(f"LR: {LR}\n")
        f.write(f"数据集: {path_dataset}\n")


# 模型加载函数
def load_model(model, load_path):
    """
    加载已保存的模型权重
    Args:
        model: 初始化好结构的FedGNN模型
        load_path: 加载路径前缀
    """
    model.load_weights(f"{load_path}_weights.h5")
    print(f"模型权重已从 {load_path}_weights.h5 加载")
    return model


# In[ ]:


if __name__ == "__main__":
    # 加载数据集
    M = load_matlab_file(path_dataset, 'M')
    Otraining = load_matlab_file(path_dataset, 'Otraining')
    Otest = load_matlab_file(path_dataset, 'Otest')
    print('There are %i interactions logs.' % np.sum(np.array(np.array(M, dtype='bool'), dtype='int32')))

    # 预处理数据    
    usernei = generate_history(Otraining)
    trainu, traini, trainlabel, train_user_index = generate_training_data(Otraining, M)
    testu, testi, testlabel = generate_test_data(Otest, M)

    # 生成公钥和私钥（用于安全通信）    
    generate_key()

    # 构建FedGNN模型
    model, userembedding_layer, itemembedding_layer = get_model(Otraining)

    # 训练模型
    user_neighbor_emb = train(model)

    # 测试模型
    test(model, user_neighbor_emb)
