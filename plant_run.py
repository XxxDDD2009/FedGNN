#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# 导入我们创建的模块
from plant_data_loader import prepare_data
from plant_model import get_federated_plant_model, get_basic_cnn_model, get_lightweight_model

# 检查GPU是否可用
print("TensorFlow版本:", tf.__version__)
print("GPU是否可用:", len(tf.config.list_physical_devices('GPU')) > 0)
print("可用的物理设备:", tf.config.list_physical_devices())
print("可用的GPU设备:", tf.config.list_physical_devices('GPU'))

# 设置GPU使用
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 设置内存增长
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        # 允许内存增长，防止一次性分配所有GPU内存
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
        print("内存增长已启用")
    except Exception as e:
        print(f"无法设置内存增长: {e}")
else:
    print("警告：未检测到GPU，将使用CPU运行（这会显著降低训练速度）")

def train_model(model, train_ds, valid_ds, epochs=20):
    """训练模型"""
    # 创建保存模型的目录
    if not os.path.exists('models'):
        os.makedirs('models')
    
    # 设置回调函数
    callbacks = [
        ModelCheckpoint('models/plant_model_best.h5', 
                        monitor='val_accuracy', 
                        save_best_only=True, 
                        mode='max', 
                        verbose=1),
        EarlyStopping(monitor='val_loss', 
                      patience=10, 
                      restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', 
                          factor=0.5, 
                          patience=5, 
                          min_lr=1e-6)
    ]
    
    # 训练模型
    history = model.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    return model, history

def plot_training_history(history):
    """绘制训练历史"""
    plt.figure(figsize=(12, 5))
    
    # 绘制准确率
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('模型准确率')
    plt.ylabel('准确率')
    plt.xlabel('周期')
    plt.legend(['训练集', '验证集'], loc='upper left')
    
    # 绘制损失
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('模型损失')
    plt.ylabel('损失')
    plt.xlabel('周期')
    plt.legend(['训练集', '验证集'], loc='upper left')
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

def evaluate_model(model, test_ds, class_names):
    """评估模型性能"""
    # 收集所有测试预测和实际标签
    all_predictions = []
    all_labels = []
    
    # 逐批次预测，避免内存问题
    for images, labels in test_ds:
        predictions = model.predict_on_batch(images)
        all_predictions.append(predictions)
        all_labels.append(labels)
    
    # 合并批次结果
    y_pred = np.vstack(all_predictions)
    y_true = np.vstack(all_labels)
    
    # 转换为类别索引
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_true, axis=1)
    
    # 计算整体准确率
    test_loss, test_acc = model.evaluate(test_ds, verbose=1)
    print(f"\n测试集准确率: {test_acc:.4f}")
    print(f"测试集损失: {test_loss:.4f}")
    
    # 打印详细的分类报告
    print("\n分类报告:")
    class_labels = [name.split('___')[-1] for name in class_names]  # 简化类别名称
    print(classification_report(y_true_classes, y_pred_classes, target_names=class_labels))
    
    # 绘制混淆矩阵 (可能需要截断，如果类别太多)
    plt.figure(figsize=(20, 20))
    max_classes_to_show = min(20, len(class_names))  # 限制显示的类别数
    if len(class_names) > max_classes_to_show:
        # 只选择前N个类别显示
        indices = np.argsort(np.bincount(y_true_classes))[-max_classes_to_show:]
        mask_true = np.isin(y_true_classes, indices)
        mask_pred = np.isin(y_pred_classes, indices)
        mask = mask_true & mask_pred
        cm = confusion_matrix(y_true_classes[mask], y_pred_classes[mask])
        selected_labels = [class_labels[i] for i in indices]
    else:
        cm = confusion_matrix(y_true_classes, y_pred_classes)
        selected_labels = class_labels
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=selected_labels, yticklabels=selected_labels)
    plt.title('混淆矩阵')
    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    
    return test_acc

def main():
    # 设置更小的批次大小，减少内存使用
    batch_size = 4
    img_size = (128, 128)  # 显著减小图像尺寸
    
    # 准备数据
    print("正在加载和准备数据...")
    train_ds, valid_ds, test_ds, class_names = prepare_data(batch_size=batch_size, img_size=img_size)
    
    # 从第一个批次获取输入形状
    for images, _ in train_ds.take(1):
        input_shape = images[0].shape
        break
    
    # 获取类别数量
    num_classes = len(class_names)
    
    # 创建模型
    print("构建模型...")
    print(f"输入形状: {input_shape}, 类别数: {num_classes}")
    
    # 使用轻量级模型以减少内存占用
    model = get_lightweight_model(input_shape=input_shape, num_classes=num_classes)
    model.summary()
    
    # 设置更少的训练周期，避免内存泄漏累积
    epochs = 10
    
    # 训练模型
    print("开始训练模型...")
    trained_model, history = train_model(
        model, train_ds, valid_ds, 
        epochs=epochs
    )
    
    # 绘制训练历史
    plot_training_history(history)
    
    # 评估模型
    print("评估模型性能...")
    test_acc = evaluate_model(trained_model, test_ds, class_names)
    
    print(f"模型训练完成! 测试集准确率: {test_acc:.4f}")
    
if __name__ == "__main__":
    # 设置较小的GPU内存块，帮助管理内存
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # 只使用部分显存
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("GPU内存增长设置成功")
        except RuntimeError as e:
            print(e)
            
    main()
