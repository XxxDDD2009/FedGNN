import numpy as np

# 生成随机批次训练数据的生成器函数
def generate_batch_data_random(batch_size,train_user_index,trainu,traini,history,trainlabel,user_neighbor_emb):
    # 获取所有用户索引
    idx = np.array(list(train_user_index.keys()))
    # 随机打乱用户顺序
    np.random.shuffle(idx)
    # 将用户分成多个批次
    batches = [idx[range(batch_size*i, min(len(idx), batch_size*(i+1)))] for i in range(len(idx)//batch_size+1) if len(range(batch_size*i, min(len(idx), batch_size*(i+1))))]

    # 无限生成器循环
    while (True):
        for i in batches:
            # 获取当前批次的所有训练样本索引
            idxs=[train_user_index[u] for u in i]
            uid=np.array([])
            iid=np.array([])
            uneiemb=user_neighbor_emb[:0]
            y=np.array([])
            # 合并所有训练样本
            for idss in idxs:
                uid=np.concatenate([uid,trainu[idss]])
                iid=np.concatenate([iid,traini[idss]])
                y=np.concatenate([y,trainlabel[idss]])
                uneiemb=np.concatenate([uneiemb,user_neighbor_emb[trainu[idss]]],axis=0)
            # 转换为整型数组
            uid=np.array(uid,dtype='int32')
            iid=np.array(iid,dtype='int32')
            # 获取用户历史交互记录
            ui=history[uid]
            # 扩展维度以匹配模型输入要求
            uid=np.expand_dims(uid,axis=1)
            iid=np.expand_dims(iid,axis=1)
            
            # 生成一个批次的数据和对应的标签
            yield ([uid,iid,ui,uneiemb], [y])


# 生成顺序批次测试数据的生成器函数
def generate_batch_data(batch_size,testu,testi,history,testlabel,user_neighbor_emb):
    # 生成索引数组
    idx = np.arange(len(testlabel))
    # 随机打乱索引
    np.random.shuffle(idx)
    y=testlabel
    # 将测试数据分成多个批次
    batches = [idx[range(batch_size*i, min(len(y), batch_size*(i+1)))] for i in range(len(y)//batch_size+1)]

    # 无限生成器循环
    while (True):
        for i in batches:
            # 准备当前批次的输入数据
            # 用户ID，扩展维度
            uid=np.expand_dims(testu[i],axis=1)
            # 物品ID，扩展维度
            iid=np.expand_dims(testi[i],axis=1)
            # 用户历史交互
            ui=history[testu[i]]
            # 用户邻居嵌入
            uneiemb=user_neighbor_emb[testu[i]]

            # 生成一个批次的数据和对应的标签
            yield ([uid,iid,ui,uneiemb], [y])
