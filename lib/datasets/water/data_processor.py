#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
水污染数据处理模块
将数据按照国家分成四个区域组，并进行预处理
输出四个区域的CSV文件
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import fontManager, FontProperties
from imblearn.over_sampling import SMOTE
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
    
    # 2. 对分类列进行独热编码,同时data_encoded也含有numeric_cols
    data_encoded = pd.get_dummies(data, columns=categorical_cols, drop_first=True)
    
    # 3. 选择特征列和目标列
    feature_cols = [col for col in data_encoded.columns if col != 'resultMeanValue']
    target_col = 'resultMeanValue'  # 将resultMeanValue作为目标列
    
    # 重新处理分组数据
    group_data = {}
    for group_name, countries in country_groups.items():
        original_indices = data[data['Country'].isin(countries)].index
        group_data[group_name] = data_encoded.loc[original_indices]
    
    # 检查和打印各组数据分布情况
    print("\n原始数据分布情况:")
    for group_name, df in group_data.items():
        print(f"组 {group_name}: {len(df)} 条记录 ({len(df)/len(data_encoded)*100:.2f}%)")
    
    # 定义数据增强函数 - SMOTE方法
    def augment_data_with_smote(df, target_size):
        """使用SMOTE算法增强数据"""
        current_size = len(df)
        if current_size >= target_size:
            return df
        
        print(f"  使用SMOTE增强数据 ({current_size} -> {target_size})")
        
        # 提取特征和标签
        X = df[feature_cols].values
        y = df[target_col].values
        
        # 创建一个虚拟分类标签用于SMOTE (SMOTE需要分类问题)
        # 将数据分为两类，以目标值的中位数为界限
        median_y = np.median(y)
        binary_y = np.where(y > median_y, 1, 0)
        
        # 计算需要增加多少各类别的样本
        class_counts = np.bincount(binary_y)
        ratios = {}
        for cls in range(len(class_counts)):
            cls_size = class_counts[cls]
            # 按比例增加每个类别的样本数
            target_cls_size = int(cls_size * (target_size / current_size))
            ratios[cls] = max(target_cls_size, cls_size)  # 确保不会减少样本数
        
        # 使用SMOTE增强数据
        try:
            smote = SMOTE(sampling_strategy=ratios, random_state=42, k_neighbors=min(5, min(class_counts)-1))
            X_resampled, y_binary_resampled = smote.fit_resample(X, binary_y)
            
            # 为每个新样本分配一个合理的目标值
            # 原始样本保持不变
            y_resampled = np.zeros(len(y_binary_resampled))
            y_resampled[:current_size] = y
            
            # 对于新生成的样本，根据其所属类别分配合理的目标值
            for i in range(current_size, len(y_resampled)):
                cls = y_binary_resampled[i]
                # 选择该类别中的一个随机样本作为基准
                original_indices = np.where(binary_y == cls)[0]
                base_idx = np.random.choice(original_indices)
                # 添加少量噪声以增加变异性
                y_resampled[i] = y[base_idx] * (1 + np.random.normal(0, 0.05))
            
            # 创建新的DataFrame
            result_df = pd.DataFrame(X_resampled, columns=feature_cols)
            result_df[target_col] = y_resampled
            return result_df
        
        except Exception as e:
            print(f"  SMOTE增强失败: {e}, 将使用带噪声的复制方法")
            return augment_data_with_noise(df, target_size)
    
    # 定义数据增强函数 - 带噪声的复制方法
    def augment_data_with_noise(df, target_size):
        """使用带噪声的复制方法增强数据"""
        current_size = len(df)
        if current_size >= target_size:
            return df
        
        print(f"  使用带噪声的复制方法增强数据 ({current_size} -> {target_size})")
        
        # 创建原始数据的副本
        result_df = df.copy()
        
        # 计算需要额外创建的样本数
        needed = target_size - current_size
        
        # 循环添加带噪声的样本
        while len(result_df) < target_size:
            # 随机抽样（允许重复）
            sample_size = min(needed, current_size)
            samples = df.sample(n=sample_size, replace=True)
            
            # 为数值特征添加少量随机噪声（标准差为原值的5%）
            for col in feature_cols:
                if pd.api.types.is_numeric_dtype(samples[col]):
                    noise = np.random.normal(0, 0.05 * samples[col].std(), size=len(samples))
                    samples[col] = samples[col] + noise
            
            # 为标签添加少量噪声
            noise = np.random.normal(0, 0.05 * samples[target_col].std(), size=len(samples))
            samples[target_col] = samples[target_col] + noise
            
            # 添加到结果中
            result_df = pd.concat([result_df, samples], ignore_index=True)
            needed = target_size - len(result_df)
        
        return result_df
    
    # 准备组列表
    groups_list = list(group_data.keys())
    
    # 计算每个客户端组需要的数据量
    clients_per_group = int(K/4)  # 每个组需要支持的客户端数量
    main_samples_per_client = maj_n  # 每个客户端主要数据量
    
    # 设定每组最小所需样本数
    min_group_size = maj_n + min_n * 3  # 主组数据 + 用于其他客户端的数据
    min_required_size = clients_per_group * main_samples_per_client  # 理论上需要的最小样本数(不重复采样)
    
    print("\n数据量需求分析:")
    print(f"- 每组分配客户端数量: {clients_per_group}")
    print(f"- 每个客户端的主组数据量: {main_samples_per_client}")
    print(f"- 理论上每组最小需要数据量(不重复): {min_required_size}")
    print(f"- 当前增强阀值设定: {min_group_size}")
    
    # 对数据量不足的组进行增强
    need_augmentation = False
    for group_name in groups_list:
        group_size = len(group_data[group_name])
        if group_size < min_group_size:
            need_augmentation = True
            print(f"\n对组 {group_name} 进行数据增强 ({group_size} -> {min_group_size})")
            # 首先尝试使用SMOTE增强
            group_data[group_name] = augment_data_with_smote(group_data[group_name], min_group_size)
        else:
            enough_for_clients = group_size / main_samples_per_client
            print(f"\n组 {group_name} 数据量充足({group_size} 条), 不需要增强, 可支持约 {enough_for_clients:.1f} 个客户端独立采样")
            if group_size < min_required_size:
                print(f"  注意: 尽管数据量超过增强阀值, 但少于理论需求({min_required_size}), 将通过重复采样填补")
    
    if not need_augmentation:
        print("\n总结: 所有组的数据量均超过当前设定的增强阀值, 不需要数据增强")
        print("  若想强制数据增强, 可提高min_group_size阀值或引入更全面的判断标准")
        print("  当前实现恢复采样方式可确保客户端数据量稳定, 但在数据不均衡的情况下可能会有样本重复")
    
    # 再次检查和打印数据分布情况
    print("\n增强后数据分布情况:")
    for group_name, df in group_data.items():
        print(f"组 {group_name}: {len(df)} 条记录")
    
    # 为每个客户端准备数据
    # 每个组负责 K/4 个客户端
    # (组列表已在前面定义)
    
    # 客户端数据分布情况记录
    client_data_stats = {}
    
    for i in range(int(K/4)):
        for g_idx, group_name in enumerate(groups_list):
            # 确定当前客户端索引
            client_idx = g_idx * int(K/4) + i
            
            # 为当前客户端收集数据
            main_group = group_name  # 主要组 (70% 的数据)
            other_groups = [g for g in groups_list if g != main_group]
            
            # 计算主组实际可用的样本数
            main_available = len(group_data[main_group])
            main_sample_size = min(main_available, maj_n)
            
            # 从主组获取数据（如果数据不足，允许重复采样）
            main_data = group_data[main_group].sample(n=main_sample_size, replace=(main_available < maj_n))
            
            # 合并所有数据
            client_data = main_data.copy()
            other_samples = {}
            
            # 从其他组获取额外数据
            for other_group in other_groups:
                other_available = len(group_data[other_group])
                other_sample_size = min(other_available, min_n)
                
                if other_sample_size > 0:
                    # 如果数据不足，允许重复采样
                    additional_data = group_data[other_group].sample(n=other_sample_size, replace=(other_available < min_n))
                    client_data = pd.concat([client_data, additional_data])
                    other_samples[other_group] = len(additional_data)
            
            # 打乱数据顺序
            client_data = client_data.sample(frac=1).reset_index(drop=True)
            
            # 记录客户端数据组成情况
            client_data_stats[client_idx] = {
                'total': len(client_data),
                'main_group': main_group,
                'main_samples': len(main_data),
                'other_samples': other_samples
            }
            
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
    
    # 打印客户端数据统计信息
    print("\n客户端数据分布情况:")
    client_sample_sizes = [stats['total'] for stats in client_data_stats.values()]
    print(f"客户端样本数量 - 平均: {np.mean(client_sample_sizes):.1f}, 最小: {np.min(client_sample_sizes)}, 最大: {np.max(client_sample_sizes)}")
    
    # 检查是否有客户端数据量小于期望值
    expected_samples = maj_n + min_n * 3 # 期望的每个客户端样本数
    below_expected = [idx for idx, stats in client_data_stats.items() if stats['total'] < expected_samples]
    if below_expected:
        print(f"\n警告: {len(below_expected)} 个客户端的样本数少于期望值 {expected_samples}:")
        for idx in below_expected[:5]:  # 只显示前5个
            stats = client_data_stats[idx]
            print(f"  客户端 {idx}: 总数 {stats['total']}, 主组({stats['main_group']}): {stats['main_samples']}, 其他组: {stats['other_samples']}")
        if len(below_expected) > 5:
            print(f"  ... 及其他 {len(below_expected)-5} 个客户端")
    
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
