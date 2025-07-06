#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.layers import BatchNormalization, Input, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2, ResNet50
from tensorflow.keras.optimizers import Adam

def get_basic_cnn_model(input_shape=(224, 224, 3), num_classes=38):
    """创建一个基础的CNN模型"""
    model = Sequential([
        Conv2D(16, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    # 编译模型
    model.compile(
        optimizer=Adam(learning_rate=0.0005),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def get_mobilenet_model(input_shape=(224, 224, 3), num_classes=38):
    """使用MobileNetV2作为基础的迁移学习模型，低内存占用版本"""
    # 加载预训练的MobileNetV2模型，不包含顶层
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape,
        alpha=0.75  # 使用更小的网络版本
    )
    
    # 冻结基础模型层
    base_model.trainable = False
    
    # 添加自定义分类层
    inputs = Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dense(64, activation='relu')(x)  # 减少神经元数量
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    # 创建最终模型
    model = Model(inputs=inputs, outputs=outputs)
    
    # 编译模型
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def get_federated_plant_model(input_shape=(224, 224, 3), num_classes=38):
    """
    创建适合联邦学习的植物疾病分类模型，低内存版本
    """
    # 减少输入图像大小以节省内存
    if input_shape[0] > 128:
        print("警告: 输入图像大小过大，可能导致内存不足。考虑在数据加载时减小图像尺寸。")
    
    model = Sequential([
        # 使用更小的过滤器数量
        Conv2D(16, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        
        # 分类层
        GlobalAveragePooling2D(),  # 使用全局平均池化而不是Flatten，大大减少参数数量
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    # 编译模型
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def get_lightweight_model(input_shape=(224, 224, 3), num_classes=38):
    """
    创建一个非常轻量级的模型，适合内存受限的环境
    """
    model = Sequential([
        # 极简卷积层
        Conv2D(8, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        MaxPooling2D(pool_size=(4, 4)),  # 大幅下采样
        
        Conv2D(16, (3, 3), activation='relu', padding='same'),
        MaxPooling2D(pool_size=(4, 4)),  # 大幅下采样
        
        # 分类层
        GlobalAveragePooling2D(),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    
    # 编译模型
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model
