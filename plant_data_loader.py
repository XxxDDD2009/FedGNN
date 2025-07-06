#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import random

def get_class_names(train_dir):
    """获取所有类别名称"""
    class_names = sorted(os.listdir(train_dir))
    return class_names

def create_image_dataset(data_dir, img_size=(224, 224), batch_size=32, shuffle=True):
    """
    创建高效的tf.data.Dataset，避免一次性加载所有图像
    
    参数:
    - data_dir: 数据目录路径
    - img_size: 图像尺寸 (宽度, 高度)
    - batch_size: 批次大小
    - shuffle: 是否打乱数据
    
    返回:
    - dataset: tf.data.Dataset对象
    - class_names: 类别名称列表
    - num_samples: 样本数量
    """
    # 检查是否为目录结构（训练集/验证集）或扁平结构（测试集）
    is_directory_structure = os.path.isdir(os.path.join(data_dir, os.listdir(data_dir)[0]))
    
    if is_directory_structure:
        # 训练集/验证集结构: 使用tf.keras.utils.image_dataset_from_directory
        dataset = tf.keras.utils.image_dataset_from_directory(
            data_dir,
            image_size=img_size,
            batch_size=batch_size,
            shuffle=shuffle,
            label_mode='categorical'
        )
        
        class_names = dataset.class_names
        # 计算样本数量
        num_samples = sum([len(files) for _, _, files in os.walk(data_dir)])
        
    else:
        # 测试集结构: 需要自定义处理
        # 从训练集获取完整的类别列表
        train_dir = os.path.join(os.path.dirname(data_dir), 'train')
        if os.path.exists(train_dir):
            class_names = sorted(os.listdir(train_dir))
        else:
            # 如果找不到训练集目录，尝试从测试集文件名中提取类别
            class_names = []  # 这需要后续处理
            
        # 创建从疾病名称到类别索引的映射
        disease_mapping = {
            'AppleScab': 'Apple___Apple_scab',
            'AppleCedarRust': 'Apple___Cedar_apple_rust',
            'CornCommonRust': 'Corn_(maize)___Common_rust_',
            'PotatoEarlyBlight': 'Potato___Early_blight',
            'PotatoHealthy': 'Potato___healthy',
            'TomatoEarlyBlight': 'Tomato___Early_blight',
            'TomatoHealthy': 'Tomato___healthy',
            'TomatoYellowCurlVirus': 'Tomato___Tomato_Yellow_Leaf_Curl_Virus'
        }
        
        # 获取所有图像文件
        image_files = [f for f in os.listdir(data_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        num_samples = len(image_files)
        
        # 这里简单处理，对于测试集，我们还是需要加载所有文件来确定其标签
        # 但我们可以优化这个过程，仅保存文件路径和标签
        file_paths = []
        labels = []
        
        for file_name in image_files:
            img_path = os.path.join(data_dir, file_name)
            
            # 从文件名提取疾病类型
            label_idx = None
            for key, class_name in disease_mapping.items():
                if key in file_name:
                    label_idx = class_names.index(class_name) if class_name in class_names else None
                    break
            
            if label_idx is not None:
                file_paths.append(img_path)
                # 创建one-hot编码标签
                label = np.zeros(len(class_names))
                label[label_idx] = 1
                labels.append(label)
        
        # 创建tf.data.Dataset
        path_ds = tf.data.Dataset.from_tensor_slices(file_paths)
        label_ds = tf.data.Dataset.from_tensor_slices(labels)
        
        # 合并路径和标签
        dataset = tf.data.Dataset.zip((path_ds, label_ds))
        
        # 定义预处理函数
        def process_path(file_path, label):
            img = tf.io.read_file(file_path)
            img = tf.image.decode_jpeg(img, channels=3)
            img = tf.image.resize(img, img_size)
            img = img / 255.0  # 归一化
            return img, label
        
        # 映射预处理函数到数据集
        dataset = dataset.map(process_path)
        
        # 批次化和优化
        if shuffle:
            dataset = dataset.shuffle(buffer_size=1000)
        dataset = dataset.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
    
    # 优化数据加载性能
    AUTOTUNE = tf.data.AUTOTUNE
    dataset = dataset.cache().prefetch(buffer_size=AUTOTUNE)
    
    return dataset, class_names, num_samples

def prepare_data(batch_size=32, img_size=(128, 128)):
    """准备训练、验证和测试数据集"""
    base_dir = './datasets/plant'
    train_dir = os.path.join(base_dir, 'train')
    valid_dir = os.path.join(base_dir, 'valid')
    test_dir = os.path.join(base_dir, 'test')
    
    # 获取类别名称
    class_names = get_class_names(train_dir)
    print(f"类别数: {len(class_names)}")
    
    # 加载训练、验证和测试数据
    print("创建训练数据集...")
    train_ds, _, train_samples = create_image_dataset(train_dir, img_size=img_size, batch_size=batch_size)
    
    print("创建验证数据集...")
    valid_ds, _, valid_samples = create_image_dataset(valid_dir, img_size=img_size, batch_size=batch_size)
    
    print("创建测试数据集...")
    test_ds, _, test_samples = create_image_dataset(test_dir, img_size=img_size, batch_size=batch_size)
    
    print(f"训练集: {train_samples} 样本")
    print(f"验证集: {valid_samples} 样本")
    print(f"测试集: {test_samples} 样本")
    
    return train_ds, valid_ds, test_ds, class_names

def visualize_batch(dataset, class_names):
    """可视化一个批次的数据样本"""
    plt.figure(figsize=(12, 12))
    
    # 获取一个批次
    for images, labels in dataset.take(1):
        for i in range(min(9, len(images))):
            plt.subplot(3, 3, i+1)
            
            # 显示图像
            plt.imshow(images[i].numpy())
            
            # 获取标签
            label_idx = np.argmax(labels[i])
            label = class_names[label_idx]
            
            plt.title(label.split('___')[-1])
            plt.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 测试数据加载函数
    train_ds, valid_ds, test_ds, class_names = prepare_data(batch_size=16)
    print("数据集创建成功!")
    
    # 可视化一个批次
    print("可视化一个批次的样本...")
    visualize_batch(train_ds, class_names)
