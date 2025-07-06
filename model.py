from keras.layers import *
from keras.models import Model
from keras import backend as K
#from keras.engine.topology import Layer, InputSpec
from keras.layers import Layer, InputSpec
#from tensorflow.python.keras.layers import Layer, InputSpec
from keras import initializers   
from keras.optimizers import *
from const import *
import keras

# 自定义掩码计算层，用于处理填充值
class ComputeMasking(keras.layers.Layer):
    def __init__(self, maskvalue=0,**kwargs):
        self.maskvalue=maskvalue
        super(ComputeMasking, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        # 创建一个掩码，将不等于maskvalue的位置设为True
        mask = K.not_equal(inputs, self.maskvalue)
        # 将掩码转换为浮点数，并乘以-99（用于注意力机制中屏蔽填充位置）
        return K.cast(mask, K.floatx())*(-99)

    def compute_output_shape(self, input_shape):
        return input_shape

# 构建FedGNN模型
def get_model(Otraining,hidden=HIDDEN,dropout=DROP):

    # 用户和物品嵌入层
    userembedding_layer = Embedding(Otraining.shape[0]+3, hidden, trainable=True)
    itemembedding_layer = Embedding(Otraining.shape[1]+3, hidden, trainable=True)

    # 输入层定义
    userid_input = Input(shape=(1,), dtype='int32')  # 用户ID输入
    itemid_input = Input(shape=(1,), dtype='int32')  # 物品ID输入
    
    ui_input = Input(shape=(HIS_LEN,), dtype='int32')  # 用户历史交互序列输入
    neighbor_embedding_input = Input(shape=(HIS_LEN,NEIGHBOR_LEN,hidden), dtype='float32')  # 用户邻居嵌入输入
    # 创建邻居掩码，用于区分有效邻居和填充
    mask_neighbor = Lambda(lambda x:K.cast(K.cast(K.sum(x,axis=-1),'bool'),'float32'))(neighbor_embedding_input)
    
    # 邻居嵌入转换
    neighbor_embeddings = TimeDistributed(TimeDistributed(Dense(hidden)))(neighbor_embedding_input)

    # 隐藏层处理
    # 将用户历史交互转换为嵌入向量
    uiemb = Dense(hidden,activation='sigmoid')(itemembedding_layer(ui_input))
    # 扩展维度以便与邻居嵌入进行注意力计算
    uiembrepeat = Lambda(lambda x :K.repeat_elements(K.expand_dims(x,axis=2),NEIGHBOR_LEN,axis=2))(uiemb) 
    # 图注意力机制实现
    attention_gat = Reshape((HIS_LEN,NEIGHBOR_LEN))(LeakyReLU()(TimeDistributed(TimeDistributed(Dense(1)))(concatenate([uiembrepeat,neighbor_embeddings]))))
    # 应用邻居掩码，将填充位置设为极小值
    attention_gat = Lambda(lambda x:x[0]+(1-x[1])*(-99))([attention_gat,mask_neighbor])
    # 聚合邻居嵌入
    agg_neighbor_embeddings = Lambda(lambda x:K.sum(K.repeat_elements(K.expand_dims(x[0],axis=3),hidden,axis=3)*x[1],axis=-2))([attention_gat,neighbor_embeddings])
    
    # 将用户历史交互嵌入与邻居聚合嵌入结合
    uiemb_agg = Dense(hidden)(concatenate([agg_neighbor_embeddings,uiemb]))
    # 用户嵌入处理
    uemb = Dense(hidden,activation='sigmoid')(Flatten()(userembedding_layer(userid_input)))
    uemb = Dropout(dropout)(uemb)  # 应用Dropout防止过拟合
    # 物品嵌入处理
    iemb = Dense(hidden,activation='sigmoid')(Flatten()(itemembedding_layer(itemid_input)))
    iemb = Dropout(dropout)(iemb)  # 应用Dropout防止过拟合
    
    # 创建历史交互掩码
    masker = ComputeMasking(Otraining.shape[1]+2)(ui_input)
    # 扩展用户嵌入维度以匹配历史长度
    uembrepeat = Lambda(lambda x :K.repeat_elements(K.expand_dims(x,axis=1),HIS_LEN,axis=1))(uemb) 

    # 计算注意力权重
    attention = Flatten()(LeakyReLU()(Dense(1)(concatenate([uembrepeat,uiemb_agg]))))
    # 应用掩码，忽略填充位置
    attention = add([attention,masker])
    # Softmax激活获取注意力权重
    attention_weight = Activation('softmax')(attention)
    # 通过注意力权重聚合历史交互信息
    uemb_g = Dot((1, 1))([uiemb, attention_weight])
    # 将聚合后的表示与用户嵌入结合
    uemb_g = Dense(hidden)(concatenate([uemb_g, uemb]))

    # 输出层，预测评分
    out = Dense(1,activation='sigmoid')(concatenate([uemb_g, iemb]))
    # 构建模型
    model = Model([userid_input,itemid_input,ui_input,neighbor_embedding_input],out)
    # 配置训练参数
    model.compile(loss=['mse'], optimizer=SGD(lr=LR,clipnorm=CLIP), metrics=['mse'])
    # 替代优化器选项
    # model.compile(loss=['mse'], optimizer=Adam(lr=LR, clipnorm=CLIP), metrics=['mse'])

    return model,userembedding_layer,itemembedding_layer