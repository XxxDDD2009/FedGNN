#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from  utils import *
from  encrypt import *
from  model import *
from preprocess import *
from expansion import *
from generator import *
from const import *
import numpy as np
import random
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 各种数据集路径选项
# path_dataset = ('/home/maxiaoming/project/FedPerGNN-main/mgcnn-master/Data/douban/training_test_dataset.mat')
# path_dataset = ('/home/maxiaoming/project/FedPerGNN-main/mgcnn-master/Data/flixster/training_test_dataset_10_NNs.mat')
#path_dataset = ('/home/maxiaoming/project/FedPerGNN-main/mgcnn-master/Data/movielens/split_1.mat')
path_dataset = ('./mgcnn-master/Data/synthetic_netflix/synthetic_netflix.mat')
# path_dataset = ('/home/maxiaoming/project/FedPerGNN-main/mgcnn-master/Data/yahoo_music/training_test_dataset_10_NNs.mat')

# In[ ]:


# 模型训练函数
def train(model):
    # 遍历每个训练轮次
    for rounds in range(EPOCH):
        # 获取当前用户嵌入矩阵
        alluserembs=userembedding_layer.get_weights()[0]
        # 执行图嵌入扩展（FedGNN的核心操作）
        user_neighbor_emb=graph_embedding_expansion(Otraining,usernei,alluserembs)
        # 生成随机批次训练数据
        traingen=generate_batch_data_random(BATCH_SIZE,train_user_index,trainu,traini,usernei,trainlabel,user_neighbor_emb)
        cnt=0
        batchloss=[]
        # 批次训练循环
        for i in traingen:
            # 获取当前模型权重
            layer_weights=model.get_weights()
            # 使用当前批次数据训练模型
            loss=model.train_on_batch(i[0],i[1])
            batchloss.append(loss)
            # 获取更新后的模型权重
            now_weights=model.get_weights()
        
            # 计算权重更新的标准差（用于差分隐私）
            sigma=np.std(now_weights[0]-layer_weights[0])
            # 添加伪交互项的噪声（差分隐私保护）
            norm=np.random.normal(0, sigma/np.sqrt(PSEUDO*BATCH_SIZE/now_weights[0].shape[0]), size=now_weights[0].shape)
            now_weights[0]+=norm
            itemembedding_layer.set_weights([now_weights[0]])
            print(np.mean(batchloss))
            # 添加本地差分隐私噪声
            for i in range(len(now_weights)):
                now_weights[i]+=np.random.laplace(0,LR*2*CLIP/np.sqrt(BATCH_SIZE)/EPS,size=now_weights[i].shape)
            model.set_weights(now_weights)
            cnt+=1
            # 每10个批次打印一次进度
            if cnt%10==0:
                print(cnt,loss)
            # 达到一轮的批次上限后停止
            if cnt==len(train_user_index)//BATCH_SIZE:
                break
        # 每轮结束后测试模型性能
        test(model, user_neighbor_emb)
    return user_neighbor_emb

# 模型测试函数
def test(model,user_neighbor_emb):            
    # 生成测试数据批次
    testgen=generate_batch_data(BATCH_SIZE,testu,testi,usernei,testlabel,user_neighbor_emb)
    # 旧版TensorFlow用predict_generator，新版用predict
    #cr = model.predict_generator(testgen, steps=len(testlabel)//BATCH_SIZE+1,verbose=1)
    cr = model.predict(testgen, steps=len(testlabel) // BATCH_SIZE + 1, verbose=1)
    # 计算并打印RMSE评估指标（均方根误差）
    print('rmse:',np.sqrt(np.mean(np.square(cr.flatten()-testlabel/LABEL_SCALE)))*LABEL_SCALE)
        
        


# In[ ]:


if __name__ == "__main__":
    
    # 加载数据集
    M = load_matlab_file(path_dataset, 'M')
    Otraining = load_matlab_file(path_dataset, 'Otraining')
    Otest = load_matlab_file(path_dataset, 'Otest')
    print('There are %i interactions logs.'%np.sum(np.array(np.array(M,dtype='bool'),dtype='int32')))

    # 预处理数据    
    usernei=generate_history(Otraining)
    trainu,traini,trainlabel,train_user_index=generate_training_data(Otraining,M)
    testu,testi,testlabel=generate_test_data(Otest,M)

    # 生成公钥和私钥（用于安全通信）    
    generate_key()

    # 构建FedGNN模型
    model,userembedding_layer,itemembedding_layer = get_model(Otraining)

    # 训练模型
    user_neighbor_emb = train(model)

    # 测试模型
    test(model,user_neighbor_emb)
