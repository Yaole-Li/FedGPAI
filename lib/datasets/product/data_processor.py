#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
产品质量数据处理模块
将数据按照房间分成四个区域组，并进行预处理
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import time

def data_processing_product(args):
    """
    根据参数加载和处理产品质量数据集
    
    Args:
        args: 包含参数的对象，例如 num_clients, num_samples 等
        
    Returns:
        X, Y: 特征和标签数据，以字典形式组织，每个客户端对应一个键
    """
    print("正在加载产品质量数据集...")
    
    # 当前脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 数据文件路径
    data_X_path = os.path.join(current_dir, 'data_X.csv')
    data_Y_path = os.path.join(current_dir, 'data_Y.csv')
    
    # 读取数据
    data_X = pd.read_csv(data_X_path)
    data_Y = pd.read_csv(data_Y_path)
    
    # 合并数据
    data = pd.merge(data_X, data_Y, on='date_time', how='inner')
    
    # 处理缺失值
    data = data.dropna()
    
    # 处理异常值
    print(f"原始数据大小: {data.shape}")
    
    # 检查目标变量是否有无穷值或NaN
    if np.any(np.isinf(data['quality'])) or np.any(np.isnan(data['quality'])):
        print("目标变量存在无穷值或NaN，进行清理")
        data = data[~np.isinf(data['quality']) & ~np.isnan(data['quality'])]
    
    # 使用IQR方法处理异常值
    Q1 = data['quality'].quantile(0.25)
    Q3 = data['quality'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    data = data[(data['quality'] >= lower_bound) & (data['quality'] <= upper_bound)]
    
    # 目标值规模调整，将quality大值缩小，避免梯度爆炸
    # 先保存原始的最大值和最小值以便后续可能的反向转换
    quality_min = data['quality'].min()
    quality_max = data['quality'].max()
    print(f"目标值范围：最小值 = {quality_min}, 最大值 = {quality_max}")
    
    # 将quality值标准化到[0,1]范围
    data['quality'] = (data['quality'] - quality_min) / (quality_max - quality_min) 
    
    print(f"异常值处理后数据大小: {data.shape}")
    
    # 提取日期列作为索引
    data['date_time'] = pd.to_datetime(data['date_time'])
    data.set_index('date_time', inplace=True)
    
    # 按房间进行分组
    # 根据传感器名称分析，共有5个房间，每个房间有3个传感器
    # T_data_X_Y 格式，其中X是房间号，Y是传感器号
    
    # 创建房间组
    # 首先获取各房间的传感器列
    room1_sensors = [col for col in data.columns if col.startswith('T_data_1_')]
    room2_sensors = [col for col in data.columns if col.startswith('T_data_2_')]
    room3_sensors = [col for col in data.columns if col.startswith('T_data_3_')]
    room4_sensors = [col for col in data.columns if col.startswith('T_data_4_')]
    room5_sensors = [col for col in data.columns if col.startswith('T_data_5_')]
    
    # 将房间5的传感器均匀分配到其他四个房间
    # 房间1得到T_data_5_1
    # 房间2得到T_data_5_2
    # 房间3得到T_data_5_3
    # 房间4得到T_data_5_3
    room1_sensors.append(room5_sensors[0])  # T_data_5_1
    room2_sensors.append(room5_sensors[1])  # T_data_5_2
    room3_sensors.append(room5_sensors[2])  # T_data_5_3
    room4_sensors.append(room5_sensors[2])  # T_data_5_3 - 分配给房间4也保证特征统一
    
    # 创建四个均衡的房间组
    room_groups = {
        'room1': room1_sensors,
        'room2': room2_sensors,
        'room3': room3_sensors,
        'room4': room4_sensors
    }
    
    # 添加公共特征(如湿度等)到每个组
    common_features = ['H_data', 'AH_data']
    for group in room_groups:
        room_groups[group] = room_groups[group] + common_features
    
    # 添加目标变量
    target_col = 'quality'
    
    # 输出各组数据统计
    for group_name, features in room_groups.items():
        print(f"{group_name}: {len(features)} 个特征")
    
    # 准备返回的数据结构
    K = args.num_clients  # 客户端数量
    maj_n = int(.7 * args.num_samples)  # 主要数据量
    min_n = int(.1 * args.num_samples)  # 次要数据量
    
    X = {}  # 特征
    Y = {}  # 标签
    
    # 准备按房间分组的数据
    group_data = {}
    for group_name, features in room_groups.items():
        # 选择当前组的特征和目标变量
        group_features = data[features]
        group_target = data[target_col]
        
        # 标准化特征数据，防止除以0的情况
        means = group_features.mean()
        stds = group_features.std()
        
        # 处理标准差为0的情况
        stds = stds.replace(0, 1)  # 将标准差为0的列替换为1，避免除以0
        
        # 应用标准化
        group_features = (group_features - means) / stds
        
        # 检查并处理标准化后的无穷值和NaN
        group_features = group_features.replace([np.inf, -np.inf], np.nan)
        group_features = group_features.fillna(0)  # 将NaN替换为0
        
        # 合并特征和目标
        group_data[group_name] = {
            'features': group_features,
            'target': group_target
        }
    
    # 检查和打印各组数据分布情况
    print("\n数据分布情况:")
    for group_name, d in group_data.items():
        print(f"组 {group_name}: {len(d['features'])} 条记录")
    
    # 将每个客户端的数据划分为：
    # 70% 来自主房间组，30% 来自其他房间组（每个10%）
    
    group_names = list(group_data.keys())  # 房间组名称列表
    
    for k in range(K):
        # 确定当前客户端的主要数据组
        main_group_idx = k % len(group_names)
        main_group = group_names[main_group_idx]
        
        # 为避免超出数据范围，确定可用的索引数量
        main_dataset_size = len(group_data[main_group]['features'])
        available_indices = min(main_dataset_size, args.num_samples)
        
        if available_indices < args.num_samples:
            print(f"警告：客户端 {k} 的主要组 {main_group} 数据不足，请考虑减少每个客户端的样本数")
        
        # 为主要组选择随机索引
        main_indices = np.random.choice(main_dataset_size, min(maj_n, available_indices), replace=False)
        
        # 从主要组中提取特征和标签
        X_main = group_data[main_group]['features'].iloc[main_indices].values
        Y_main = group_data[main_group]['target'].iloc[main_indices].values
        # 调整Y的维度为[n,1]，以匹配模型输出维度
        Y_main = Y_main.reshape(-1, 1)
        
        # 初始化特征和标签数组
        X[k] = X_main
        Y[k] = Y_main
        
        # 从其他组中获取数据
        other_groups = [g for g in group_names if g != main_group]
        
        for i, other_group in enumerate(other_groups):
            other_dataset_size = len(group_data[other_group]['features'])
            
            # 选择索引（确保不超出范围）
            other_indices = np.random.choice(other_dataset_size, min(min_n, other_dataset_size), replace=False)
            
            # 从其他组中提取特征和标签
            X_other = group_data[other_group]['features'].iloc[other_indices].values
            Y_other = group_data[other_group]['target'].iloc[other_indices].values
            # 调整Y的维度为[n,1]，以匹配模型输出维度
            Y_other = Y_other.reshape(-1, 1)
            
            # 整合到客户端数据中
            if len(X_other) > 0:
                X[k] = np.vstack((X[k], X_other))
                Y[k] = np.concatenate((Y[k], Y_other))
        
        # 确保数据量不超过指定的样本数
        if len(X[k]) > args.num_samples:
            # 随机选择指定数量的样本
            selected_indices = np.random.choice(len(X[k]), args.num_samples, replace=False)
            X[k] = X[k][selected_indices]
            Y[k] = Y[k][selected_indices]
    
    print(f"数据处理完成，共 {K} 个客户端，每个客户端约 {args.num_samples} 个样本")
    
    return X, Y

if __name__ == "__main__":
    # 测试代码
    class Args:
        def __init__(self):
            self.num_clients = 10
            self.num_samples = 1000
    
    args = Args()
    X, Y = data_processing_product(args)
    
    # 统计分析
    for k in range(args.num_clients):
        print(f"客户端 {k}: {X[k].shape[0]} 样本, 特征维度 {X[k].shape[1]}")
