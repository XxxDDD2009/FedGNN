from  encrypt import *
from const import *
import random
import numpy as np

def graph_embedding_expansion(Otraining,usernei,alluserembs):
    # 本地加密过程
    local_ciphertext = []
    for i in tqdm(usernei):
        messages = []
        for j in i:
            # 对非填充项进行加密
            if j!= Otraining.shape[1]+2:
                messages.append(base64.b64encode(sign(str(j))).decode('utf-8'))
        local_ciphertext.append(messages)
        
    # 构建本地ID和密文的映射字典
    local_mapping_dict = {base64.b64encode(sign(str(j))).decode('utf-8'):j for j in range(Otraining.shape[1]+3)}
    
    # 假设本地密文已发送至第三方服务器
    # 以下是服务器端的处理逻辑

    # 构建密文到用户ID的映射字典
    cipher2userid = {}
    for userid,i in enumerate(local_ciphertext):
        for j in i:
            if j not in cipher2userid:
                cipher2userid[j] = [userid]
            else:
                cipher2userid[j].append(userid)

    # 第三方服务器准备数据
    # 为每个用户收集邻居信息
    send_data = []
    for userid,i in tqdm(enumerate(local_ciphertext)):
        neighbor_info={}
        for j in i:
            # 获取共同交互项的用户嵌入向量
            neighbor_id = [alluserembs[uid] for uid in cipher2userid[j]]
            if len(neighbor_id):
                neighbor_info[j] = neighbor_id
        send_data.append(neighbor_info)
        
    # 第三方服务器分发send_data到各个本地客户端
    
    # 本地客户端进行图扩展处理
    user_neighbor_emb = []
    for userid,user_items in tqdm(enumerate(usernei)):
        # 接收来自服务器的数据
        receive_data = send_data[userid]
        # 解密数据，将密文映射回实际物品ID
        decrypted_data = {local_mapping_dict[item_key]:receive_data[item_key] for item_key in receive_data}
        all_neighbor_embs=[]
        for item in user_items:
            if item in decrypted_data:
                # 获取该物品的邻居用户嵌入
                neighbor_embs = decrypted_data[item]
                # 随机打乱顺序以增强隐私保护
                random.shuffle(neighbor_embs)
                # 限制邻居数量
                neighbor_embs = neighbor_embs[:NEIGHBOR_LEN] 
                # 如果邻居数不足，用零向量填充
                neighbor_embs += [[0.]*HIDDEN]*(NEIGHBOR_LEN-len(neighbor_embs))
            else:
                # 如果没有邻居信息，使用全零向量
                neighbor_embs = [[0.]*HIDDEN]*NEIGHBOR_LEN
            all_neighbor_embs.append(neighbor_embs)
        # 限制历史交互项数量
        all_neighbor_embs = all_neighbor_embs[:HIS_LEN]
        # 如果历史交互项不足，用零矩阵填充
        all_neighbor_embs += [[[0.]*HIDDEN]*NEIGHBOR_LEN]*(HIS_LEN-len(all_neighbor_embs))
        user_neighbor_emb.append(all_neighbor_embs)
    
    # 转换为numpy数组并返回
    user_neighbor_emb = np.array(user_neighbor_emb,dtype='float32')
    return user_neighbor_emb
    
