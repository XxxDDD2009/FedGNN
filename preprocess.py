import random
import numpy as np
from const import *

# 生成用户历史交互记录
def generate_history(Otraining):
    # 构建用户历史交互列表
    history=[]
    for i in range(Otraining.shape[0]):
        user_history=[]
        # 遍历用户的所有交互项目
        for j in range(len(Otraining[i])):
            # 如果有交互（评分不为0），加入历史记录
            if Otraining[i][j]!=0.0:
                user_history.append(j)
        # 将序列进行随机排列
        random.shuffle(user_history)

        # 限制历史记录长度，不足则用填充值补齐
        user_history=user_history[:HIS_LEN]    
        history.append(user_history+[Otraining.shape[1]+2]*(HIS_LEN-len(user_history)))
    # 转换为numpy数组
    history=np.array(history,dtype='int32')
    return history

# 生成训练数据
def generate_training_data(Otraining,M):
    # 构建训练用的用户-物品对
    trainu=[]  # 用户索引
    traini=[]  # 物品索引
    trainlabel=[]  # 交互标签（评分）
    train_user_index={}  # 每个用户的训练样本索引
    for i in range(Otraining.shape[0]):
        user_index=[]
        for j in range(len(Otraining[i])):
            # 如果用户与物品有交互
            if Otraining[i][j]!=0:
                user_index.append(len(trainu))
                trainu.append(i)
                traini.append(j)
                # 归一化评分值
                trainlabel.append(M[i][j]/LABEL_SCALE)
        # 记录用户的训练样本索引
        if len(user_index):
            train_user_index[i]=user_index
            
    # 转换为numpy数组
    trainu=np.array(trainu,dtype='int32')
    traini=np.array(traini,dtype='int32')
    trainlabel=np.array(trainlabel,dtype='int32')
    return trainu,traini,trainlabel,train_user_index

# 生成测试数据
def generate_test_data(Otest,M):
    # 构建测试用的用户-物品对
    testu=[]  # 用户索引
    testi=[]  # 物品索引
    testlabel=[]  # 交互标签（评分）
    
    for i in range(Otest.shape[0]):
        for j in range(len(Otest[i])):
            # 如果用户与物品有交互
            if Otest[i][j]!=0:
                testu.append(i)
                testi.append(j)
                # 归一化评分值
                testlabel.append(M[i][j]/LABEL_SCALE)
    
    # 转换为numpy数组
    testu=np.array(testu,dtype='int32')
    testi=np.array(testi,dtype='int32')
    testlabel=np.array(testlabel,dtype='int32')
    return testu,testi,testlabel
