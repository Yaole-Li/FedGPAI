#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
水污染数据处理模块
将数据按照国家分成四个区域组，并进行预处理
输出四个区域的CSV文件
"""

import os
import numpy as np
import pandas as pd
from numpy import linalg as LA
from collections import defaultdict
import time

def data_processing_water(args):
    """
    根据参数加载和处理水污染数据集
    
    Args:
        args: 包含参数的对象，例如 num_clients, num_samples 等
        
    Returns:
        X, Y: 特征和标签数据，以字典形式组织，每个客户端对应一个键
    """
    print("正在加载水污染数据集...")
    
    # 当前脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 数据文件路径
    data_path = os.path.join(current_dir, 'waterPollution.csv')
    
    # 读取数据
    data = pd.read_csv(data_path)
    
    # 处理缺失值
    data = data.dropna()
    
    # 地理分组：基于欧洲地理区域分组
    # 组1：西欧 - 法国, 比利时, 卢森堡, 荷兰
    # 组2：英国和北欧 - 英国, 爱尔兰, 挪威, 瑞典, 芬兰, 丹麦
    # 组3：中欧 - 德国, 奥地利, 瑞士, 波兰, 捷克, 斯洛伐克
    # 组4：南欧和东欧 - 西班牙, 意大利, 葡萄牙, 保加利亚, 罗马尼亚, 立陶宛, 拉脱维亚, 塞尔维亚, 克罗地亚, 乌克兰, 白俄罗斯, 俄罗斯
    
    country_groups = {
        'western_europe': ['France', 'Belgium', 'Luxembourg', 'Netherlands'],
        'northern_europe': ['United Kingdom', 'Ireland', 'Norway', 'Sweden', 'Finland', 'Denmark'],
        'central_europe': ['Germany', 'Austria', 'Switzerland', 'Poland', 'Czech Republic', 'Slovakia'],
        'southern_eastern_europe': ['Spain', 'Italy', 'Portugal', 'Bulgaria', 'Romania', 'Lithuania', 
                                  'Latvia', 'Serbia', 'Croatia', 'Ukraine', 'Belarus', 'Russia']
    }
    
    # 按照组分割数据
    group_data = {}
    for group_name, countries in country_groups.items():
        group_data[group_name] = data[data['Country'].isin(countries)]
    
    # 输出各组数据统计
    for group_name, df in group_data.items():
        print(f"{group_name}: {len(df)} 条记录，占比 {len(df)/len(data)*100:.2f}%")
    
    # 准备返回的数据结构
    K = args.num_clients  # 客户端数量
    maj_n = int(.7 * args.num_samples)  # 主要数据量
    min_n = int(.1 * args.num_samples)  # 次要数据量
    
    X = {}  # 特征
    Y = {}  # 标签
    
    # 识别和处理非数值列
    # 1. 选择有意义的特征列 (排除 ID、类别等不适合数值计算的列)
    numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = [col for col in data.columns if col not in numeric_cols and col != 'waste_treatment_recycling_percent']
    
    print(f"数值列数量: {len(numeric_cols)}, 分类列数量: {len(categorical_cols)}")
    
    # 2. 对分类列进行独热编码
    data_encoded = pd.get_dummies(data, columns=categorical_cols, drop_first=True)
    
    # 3. 选择特征列和目标列
    feature_cols = [col for col in data_encoded.columns if col != 'resultMeanValue']
    target_col = 'resultMeanValue'  # 将resultMeanValue作为目标列
    
    # 重新处理分组数据
    group_data = {}
    for group_name, countries in country_groups.items():
        original_indices = data[data['Country'].isin(countries)].index
        group_data[group_name] = data_encoded.loc[original_indices]
    
    # 为每个客户端准备数据
    # 每个组负责 K/4 个客户端
    groups_list = list(group_data.keys())
    
    for i in range(int(K/4)):
        for g_idx, group_name in enumerate(groups_list):
            # 确定当前客户端索引
            client_idx = g_idx * int(K/4) + i
            
            # 为当前客户端收集数据
            main_group = group_name  # 主要组 (70% 的数据)
            # 其他三个组各提供 10% 的数据
            other_groups = [g for g in groups_list if g != main_group]
            
            # 从主组获取数据
            main_data = group_data[main_group].sample(maj_n) if len(group_data[main_group]) > maj_n else group_data[main_group]
            
            # 合并所有数据
            client_data = main_data.copy()
            
            # 从其他组获取额外数据
            for other_group in other_groups:
                if len(group_data[other_group]) >= min_n:
                    additional_data = group_data[other_group].sample(min_n)
                    client_data = pd.concat([client_data, additional_data])
            
            # 打乱数据顺序
            client_data = client_data.sample(frac=1).reset_index(drop=True)
            
            # 分离特征和标签
            # 确保数据可以转换为浮点数
            try:
                X[client_idx] = client_data[feature_cols].values.astype('float32')
                Y[client_idx] = client_data[target_col].values.astype('float32')
            except ValueError as e:
                print(f"转换错误: {e}")
                # 调试信息
                non_numeric = [col for col in feature_cols if not pd.api.types.is_numeric_dtype(client_data[col])]
                if non_numeric:
                    print(f"发现非数值列: {non_numeric}")
                raise
    
    # 计算归一化参数
    X_norm_max_clients = np.zeros((1, K))
    Y_max_clients = np.zeros((1, K))
    Y_min_clients = np.zeros((1, K))
    
    for i in range(K):
        # 计算每个样本的范数
        X_n = np.zeros((1, len(X[i])))
        for j in range(len(X[i])):
            X_n[0, j] = LA.norm(X[i][j, :])
        
        X_norm_max_clients[0, i] = np.max(X_n)
        Y_max_clients[0, i] = np.max(Y[i])
        Y_min_clients[0, i] = np.min(Y[i])
    
    # 全局归一化参数
    X_norm_max = np.max(X_norm_max_clients)
    Y_max = np.max(Y_max_clients)
    Y_min = np.min(Y_min_clients)
    
    # 归一化特征和标签
    for i in range(K):
        for j in range(len(X[i])):
            X[i][j, :] = X[i][j, :] / X_norm_max
        Y[i] = (Y[i] - Y_min) / (Y_max - Y_min)
    
    print(f"数据处理完成，共 {K} 个客户端，每个客户端约 {args.num_samples} 条数据")
    
    return X, Y

def analyze_country_distribution():
    """输出每个国家组中的数据分布情况"""
    # 当前脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 数据文件路径
    data_path = os.path.join(current_dir, 'waterPollution.csv')
    
    # 读取数据
    data = pd.read_csv(data_path)
    
    # 处理缺失值
    data = data.dropna()
    
    # 地理分组
    country_groups = {
        '组1-西欧': ['France', 'Belgium', 'Luxembourg', 'Netherlands'],
        '组2-英国和北欧': ['United Kingdom', 'Ireland', 'Norway', 'Sweden', 'Finland', 'Denmark'],
        '组3-中欧': ['Germany', 'Austria', 'Switzerland', 'Poland', 'Czech Republic', 'Slovakia'],
        '组4-南欧和东欧': ['Spain', 'Italy', 'Portugal', 'Bulgaria', 'Romania', 'Lithuania', 
                      'Latvia', 'Serbia', 'Croatia', 'Ukraine', 'Belarus', 'Russia']
    }
    
    # 统计每个国家的数据量
    country_counts = data['Country'].value_counts()
    
    # 输出分组信息
    print("\n地理位置分组情况:")
    print("-" * 60)
    
    total_rows = len(data)
    group_totals = defaultdict(int)
    
    for group_name, countries in country_groups.items():
        print(f"\n{group_name}:")
        print("-" * 40)
        print(f"{'国家':<20} {'数量':<10} {'百分比':<10}")
        print("-" * 40)
        
        for country in countries:
            if country in country_counts:
                count = country_counts[country]
                percentage = (count / total_rows) * 100
                print(f"{country:<20} {count:<10} {percentage:.2f}%")
                group_totals[group_name] += count
            else:
                print(f"{country:<20} 0          0.00%")
    
    print("\n各组汇总:")
    print("-" * 40)
    print(f"{'组名':<20} {'总数量':<10} {'总百分比':<10}")
    print("-" * 40)
    for group_name, total in group_totals.items():
        percentage = (total / total_rows) * 100
        print(f"{group_name:<20} {total:<10} {percentage:.2f}%")
        
    return data, country_groups

def split_and_save_by_region():
    """将数据按地理区域分割并保存为独立的CSV文件"""
    # 获取分析结果
    data, country_groups = analyze_country_distribution()
    
    # 当前脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 英文区域名映射
    region_names = {
        '组1-西欧': 'western_europe',
        '组2-英国和北欧': 'northern_europe',
        '组3-中欧': 'central_europe',
        '组4-南欧和东欧': 'southern_eastern_europe'
    }
    
    # 为每个地区创建并保存CSV文件
    for group_name, countries in country_groups.items():
        region_name = region_names[group_name]
        region_data = data[data['Country'].isin(countries)]
        
        # 输出文件路径
        output_file = os.path.join(current_dir, f"{region_name}_water_pollution.csv")
        
        # 保存CSV文件
        region_data.to_csv(output_file, index=False)
        
        print(f"已保存 {group_name} 的数据到 {output_file}, 共 {len(region_data)} 条记录")

if __name__ == "__main__":
    # 分析国家分布情况并分割保存CSV文件
    start_time = time.time()
    split_and_save_by_region()
    end_time = time.time()
    
    print(f"\n处理完成，耗时: {end_time - start_time:.2f} 秒")
