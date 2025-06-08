import os
import numpy as np
import pandas as pd
import glob
from numpy import linalg as LA
from sklearn.model_selection import train_test_split
import torch  # 导入PyTorch，用于类型转换

def data_processing_power(args):
    """
    处理电力数据集，按日期季节分成四类（春夏秋冬）
    参数:
        args: 包含数据处理参数的对象
    返回:
        X: 客户端特征数据列表
        Y: 客户端目标数据列表
    """
    print("正在加载电力数据集...")
    
    # 数据路径
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)))
    csv_files = sorted(glob.glob(os.path.join(data_path, "*.csv")))
    
    if len(csv_files) == 0:
        raise FileNotFoundError(f"在{data_path}中未找到CSV文件")
    
    # 按季节将文件分组
    spring_files = [f for f in csv_files if any(month in os.path.basename(f) for month in ['Mar', 'Apr', 'May'])]
    summer_files = [f for f in csv_files if any(month in os.path.basename(f) for month in ['Jun', 'Jul', 'Aug'])]
    autumn_files = [f for f in csv_files if any(month in os.path.basename(f) for month in ['Sep', 'Oct', 'Nov'])]
    winter_files = [f for f in csv_files if any(month in os.path.basename(f) for month in ['Dec', 'Jan', 'Feb'])]
    
    # 打印各季节文件数
    print(f"春季文件数: {len(spring_files)}")
    print(f"夏季文件数: {len(summer_files)}")
    print(f"秋季文件数: {len(autumn_files)}")
    print(f"冬季文件数: {len(winter_files)}")
    
    # 目标变量名称
    target_col = 'Battery_Active_Power'
    
    # 准备每个季节的数据
    group_data = {}
    
    for group_name, files in [
        ('spring', spring_files[:1]),  # 每个季节只处理1个文件，避免内存问题
        ('summer', summer_files[:1]),
        ('autumn', autumn_files[:1]),
        ('winter', winter_files[:1])
    ]:
        if not files:
            print(f"警告: {group_name} 季节没有数据文件")
            continue
            
        # 加载该季节的文件
        print(f"\n处理 {group_name} 季节数据:")
        for file in files:
            print(f"  加载文件: {os.path.basename(file)}")
            try:
                # 读取CSV文件
                df = pd.read_csv(file)
                print(f"  原始数据形状: {df.shape}")
                
                # 检查目标列是否存在
                if target_col not in df.columns:
                    print(f"  警告: 文件 {os.path.basename(file)} 中没有目标列 {target_col}")
                    continue
                
                # 移除Timestamp列，避免日期解析问题
                if 'Timestamp' in df.columns:
                    df = df.drop('Timestamp', axis=1)
                
                # 处理缺失值
                df = df.dropna()
                print(f"  缺失值处理后形状: {df.shape}")
                
                # 处理异常值
                if np.any(np.isinf(df[target_col])) or np.any(np.isnan(df[target_col])):
                    print(f"  目标变量存在无穷值或NaN，进行清理")
                    df = df[~np.isinf(df[target_col]) & ~np.isnan(df[target_col])]
                
                # 使用IQR方法处理异常值
                Q1 = df[target_col].quantile(0.25)
                Q3 = df[target_col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df = df[(df[target_col] >= lower_bound) & (df[target_col] <= upper_bound)]
                print(f"  异常值处理后形状: {df.shape}")
                
                # 准备特征和目标
                features = [col for col in df.columns if col != target_col]
                
                if len(features) == 0:
                    print(f"  警告: 没有找到有效特征")
                    continue
                
                # 提取特征和目标
                X_features = df[features].values  # 直接转换为numpy数组
                Y_target = df[target_col].values  # 直接转换为numpy数组
                
                # 存储到组数据中
                group_data[group_name] = {
                    'features': X_features,
                    'target': Y_target,
                    'feature_names': features
                }
                print(f"  处理完成，特征数: {len(features)}, 样本数: {len(Y_target)}")
                
            except Exception as e:
                print(f"  处理文件 {os.path.basename(file)} 时出错: {str(e)}")
                continue
    
    # 检查是否有有效的组数据
    if not group_data:
        raise ValueError("没有有效的数据可以处理")
    
    # 检查和打印各组数据分布情况
    print("\n数据分布情况:")
    for group_name, d in group_data.items():
        print(f"组 {group_name}: {d['features'].shape[0]} 条记录, {d['features'].shape[1]} 个特征")
    
    # 为每个客户端分配数据
    K = args.num_clients  # 客户端数量
    X = [None] * K  # 客户端特征
    Y = [None] * K  # 客户端标签
    
    # 限制每个客户端的样本数
    samples_per_client = min(args.num_samples, min([d['features'].shape[0] for d in group_data.values()]))
    maj_n = int(samples_per_client * 0.7)  # 主要组的样本数量
    min_n = samples_per_client - maj_n  # 次要组的样本数量
    
    print(f"每个客户端样本数: {samples_per_client} (主要组: {maj_n}, 次要组: {min_n})")
    
    # 准备样本数据
    group_names = list(group_data.keys())  # 组名称列表
    
    # 创建客户端数据（纯NumPy数组）
    for i in range(K):
        # 确定当前客户端的主要数据组
        main_group_idx = i % len(group_names)
        main_group = group_names[main_group_idx]
        
        # 提取主要组数据
        main_data = group_data[main_group]
        main_dataset_size = main_data['features'].shape[0]
        main_samples = min(maj_n, main_dataset_size)
        
        # 随机选择索引
        main_indices = np.random.choice(main_dataset_size, main_samples, replace=False)
        
        # 提取主要组的特征和标签
        X[i] = main_data['features'][main_indices]
        Y[i] = main_data['target'][main_indices]
        
        # 为次要组计算每组样本数量
        other_groups = [g for g in group_names if g != main_group]
        if other_groups:  # 确保有其他组
            samples_per_other_group = min_n // len(other_groups)
            
            for other_group in other_groups:
                # 提取次要组数据
                other_data = group_data[other_group]
                other_dataset_size = other_data['features'].shape[0]
                other_samples = min(samples_per_other_group, other_dataset_size)
                
                if other_samples > 0:
                    # 随机选择索引
                    other_indices = np.random.choice(other_dataset_size, other_samples, replace=False)
                    
                    # 提取次要组的特征和标签
                    X_other = other_data['features'][other_indices]
                    Y_other = other_data['target'][other_indices]
                    
                    # 整合到客户端数据中
                    X[i] = np.vstack((X[i], X_other))
                    Y[i] = np.append(Y[i], Y_other)
    
    # 全局标准化处理，参考WEC数据处理方法
    X_norm_max_clients = np.zeros((1, K))
    Y_max_clients = np.zeros((1, K))
    Y_min_clients = np.zeros((1, K))
    
    # 计算每个客户端的特征范数最大值和标签最大最小值
    for i in range(K):
        n_samples = X[i].shape[0]
        X_n = np.zeros((1, n_samples))
        for j in range(n_samples):
            X_n[0, j] = LA.norm(X[i][j, :])
        X_norm_max_clients[0, i] = np.max(X_n)
        Y_max_clients[0, i] = np.max(Y[i])
        Y_min_clients[0, i] = np.min(Y[i])
    
    # 全局最大范数和标签边界
    X_norm_max = np.max(X_norm_max_clients)
    Y_max = np.max(Y_max_clients)
    Y_min = np.min(Y_min_clients)
    
    # 应用全局标准化
    for i in range(K):
        # 标准化特征
        for j in range(X[i].shape[0]):
            X[i][j, :] = X[i][j, :] / X_norm_max
        
        # 标准化标签到[0,1]区间
        Y[i] = (Y[i] - Y_min) / (Y_max - Y_min)
        
        # 调整Y的维度为[n,1]，以匹配模型输出维度
        Y[i] = Y[i].reshape(-1, 1)
    
    print(f"数据处理完成，共 {K} 个客户端，每个客户端约 {samples_per_client} 个样本")
    
    # 特别注意：这里已经是NumPy数组，在FedGPAI算法开始处理前会被移到CUDA设备上。
    return X, Y
