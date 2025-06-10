import numpy as np
import argparse
import torch
import os
import gc
import time
import matplotlib.pyplot as plt
from lib.datasets.data_loader import data_loader
from lib.FedGPAI.models import MLPFeatureExtractor, MLPRegressor
from lib.FedGPAI.FedGPAI_advanced import FedGPAI_advanced
from copy import deepcopy

parser = argparse.ArgumentParser()

# 数据集和任务相关参数
parser.add_argument("--dataset", default='WEC', type=str, help="数据集名称")
parser.add_argument("--task", default='regression', type=str, help="任务类型")

# 客户端相关参数
parser.add_argument("--num_clients", default=100, type=int, help="客户端数量")
parser.add_argument("--num_samples", default=250, type=int, help="每个客户端的样本数量")
parser.add_argument("--test_ratio", default=0.2, type=float, help="测试集比例")

# 模型相关参数
parser.add_argument("--regularizer", default=1e-6, type=float, help="正则化参数")
parser.add_argument("--global_rounds", default=20, type=int, help="训练轮数")
parser.add_argument("--local_rounds", default=5, type=int, help="本地训练轮数")

# MLP模型参数
parser.add_argument("--extractor_hidden_dims", default="256,128,64", type=str, help="特征提取器MLP隐藏层维度")
parser.add_argument("--regressor_hidden_dims", default="32,16", type=str, help="回归器MLP隐藏层维度")
parser.add_argument("--output_dim", default=32, type=int, help="特征提取器输出维度/回归器输入维度")

args = parser.parse_args()

# 设置学习率
args.eta = 1/np.sqrt(args.num_samples)

# 解析隐藏层维度字符串
def parse_hidden_dims(hidden_dims_str):
    """将逗号分隔的字符串转换为整数列表"""
    if not hidden_dims_str:
        return []
    return [int(dim) for dim in hidden_dims_str.split(',')]

# 加载数据集
print(f"正在加载 {args.dataset} 数据集...")
X, Y = data_loader(args)

# 获取数据维度
M, N = X[0].shape
K = args.num_clients
M *= K

# 解析MLP模型隐藏层维度
extractor_hidden_dims = parse_hidden_dims(args.extractor_hidden_dims)
regressor_hidden_dims = parse_hidden_dims(args.regressor_hidden_dims)

# 初始化设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

print(f"\n创建MLP模型 - 输入维度: {N}")
print(f"  特征提取器: 隐藏层{extractor_hidden_dims}, 输出维度{args.output_dim}")
print(f"  回归器: 隐藏层{regressor_hidden_dims}, 输出维度 1")

# 创建保存模型的目录
# 使用方法名称_数据集_客户端数量_全局联邦训练轮数_时间戳作为文件夹名称
current_time = time.strftime('%Y%m%d_%H%M%S')
checkpoint_dir = os.path.join("checkpoints", f"Local_{args.dataset}_{args.num_clients}_{args.global_rounds}_{current_time}")
os.makedirs(checkpoint_dir, exist_ok=True)
print(f"检查点将保存到: {checkpoint_dir}")

# 创建日志文件
log_file_name = f"Local_{args.dataset}_{args.num_clients}_{args.global_rounds}.txt"
log_file_path = os.path.join(checkpoint_dir, log_file_name)

# 记录训练起始信息到日志
with open(log_file_path, 'w') as log_file:
    log_file.write(f"===== 训练开始 =====\n")
    log_file.write(f"方法: Local\n")
    log_file.write(f"数据集: {args.dataset}\n")
    log_file.write(f"客户端数量: {args.num_clients}\n")
    log_file.write(f"全局训练轮数: {args.global_rounds}\n")
    log_file.write(f"时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

print(f"日志将保存到: {log_file_path}")

# 初始化性能评估指标
mse_local = [torch.zeros((args.num_samples, 1), dtype=torch.float32).to(device) for _ in range(K)]
m_local = [torch.zeros((args.num_samples, args.global_rounds), dtype=torch.float32).to(device) for _ in range(K)]

# 跟踪训练过程中的MSE和MAE
mse_history = []
mae_history = []
rounds_history = []

# 跟踪最小值
best_mse = float('inf')
best_mae = float('inf')

print(f"开始本地训练 (每个客户端独立训练 {args.global_rounds} 轮)...")

# 创建每个客户端的独立 MLP 模型
local_models = []
for i in range(K):
    # 初始化客户端模型
    model = FedGPAI_advanced(
        lam=args.regularizer, 
        rf_feature=N,  # 输入维度
        eta=args.eta, 
        regressor_type='mlp', 
        extractor_hidden_dims=extractor_hidden_dims,
        regressor_hidden_dims=regressor_hidden_dims,
        output_dim=args.output_dim,
        num_clients=args.num_clients, 
        is_global=False,
    )
    local_models.append(model)

# 执行本地训练过程 - 每个客户端完全独立
for cc in range(args.global_rounds):
    print(f"\n轮次 {cc+1}/{args.global_rounds}")
    
    # 创建误差记录器
    e_local = [torch.zeros((args.num_samples, 1), dtype=torch.float32).to(device) for _ in range(K)]
    
    # 每个客户端独立训练
    for j in range(K):
        if j % 10 == 0:
            print(f"  训练客户端 {j+1}/{K}")
        
        # 为当前客户端设置优化器 - 同时优化特征提取器和回归器参数
        params = list(local_models[j].feature_extractor.parameters()) + list(local_models[j].regressor.parameters())
        optimizer = torch.optim.Adam(params, lr=args.eta)
        loss_fn = torch.nn.MSELoss()
        
        # 执行本地训练回合
        for _ in range(args.local_rounds):
            batch_losses = []
            
            # 样本级训练
            for i in range(args.num_samples):
                # 将NumPy数组转换为PyTorch张量并移动到指定设备
                x_j = torch.tensor(X[j][i:i+1, :], dtype=torch.float32).to(device)
                y_j = torch.tensor(Y[j][i], dtype=torch.float32).to(device)
                
                # 清除之前的梯度
                optimizer.zero_grad()
                
                # 前向传播
                output = local_models[j].forward(x_j)
                
                # 计算损失
                loss = loss_fn(output, y_j.view(-1, 1))
                batch_losses.append(loss.item())
                
                # 反向传播
                loss.backward()
                
                # 更新参数
                optimizer.step()
        
        # 计算并记录损失
        for i in range(args.num_samples):
            # 将NumPy数组转换为PyTorch张量并移动到指定设备
            x_j = torch.tensor(X[j][i:i+1, :], dtype=torch.float32).to(device)
            y_j = torch.tensor(Y[j][i], dtype=torch.float32).to(device)
            
            # 仅进行前向传播计算损失，不进行反向传播
            with torch.no_grad():
                output = local_models[j].forward(x_j)
                mse = ((output - y_j.view(-1, 1)) ** 2)
            
            # 记录误差
            m_local[j][i, cc] = mse.item()
            
            # 计算累积误差
            if i == 0:
                e_local[j][i, 0] = mse.item()
            else:
                e_local[j][i, 0] = (1/(i+1)) * ((i*e_local[j][i-1, 0]) + mse.item())
    
    # 计算平均误差
    for j in range(K):
        mse_local[j] = (1/(cc+1)) * ((cc*mse_local[j]) + e_local[j])
    
    # 计算所有客户端的平均MSE，用于显示
    avg_mse = torch.mean(torch.stack([mse[args.num_samples-1, 0] for mse in mse_local]))
    
    # 每轮记录MSE和MAE
    current_mse = avg_mse.item()
    current_mae = torch.mean(torch.sqrt(avg_mse)).item()
    mse_history.append(current_mse)
    mae_history.append(current_mae)
    rounds_history.append(cc+1)
    
    # 更新最小值
    best_mse = min(best_mse, current_mse)
    best_mae = min(best_mae, current_mae)
    
    # 每5轮计算并输出一次MAE和MSE
    if (cc+1) % 5 == 0 or cc == 0:
        print(f"\n  Round {cc+1} - MSE: {current_mse:.6f}, MAE: {current_mae:.6f}")
        
        # 将结果写入日志文件
        with open(log_file_path, 'a') as log_file:
            log_file.write(f"轮次 {cc+1}/{args.global_rounds}\n")
            log_file.write(f"MSE: {current_mse:.6f}\n")
            log_file.write(f"MAE: {current_mae:.6f}\n")
            log_file.write(f"时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # 保存模型 - 分别保存特征提取器和回归器状态
        saved_models = []
        for model in local_models:
            saved_model = {
                'feature_extractor': model.feature_extractor.state_dict(),
                'regressor': model.regressor.state_dict()
            }
            saved_models.append(saved_model)
            
        checkpoint = {
            'epoch': cc + 1,
            'local_models': saved_models,
            'mse': avg_mse.item(),
            'mae': current_mae
        }
        checkpoint_filename = f"epoch_{cc+1}.pt"
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_filename)
        torch.save(checkpoint, checkpoint_path)
        print(f"  模型已保存到: {checkpoint_path}")
        print(f"  结果已记录到: {log_file_path}")
    
    # 清理显存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

# 计算每个客户端的最终MSE
final_mse_per_client = [mse[args.num_samples-1, 0].item() for mse in mse_local]
final_mse = np.mean(final_mse_per_client)
final_std = np.std(final_mse_per_client)

# 绘制和保存MSE曲线
plt.figure(figsize=(10, 6))
plt.plot(rounds_history, mse_history, 'b-o')
plt.xlabel('Round')
plt.ylabel('Mean Squared Error (MSE)')
plt.title('Local Training MSE over Rounds')
plt.grid(True)
plt.tight_layout()
mse_plot_path = os.path.join(checkpoint_dir, 'mse_curve.png')
plt.savefig(mse_plot_path)
plt.close()

# 绘制和保存MAE曲线
plt.figure(figsize=(10, 6))
plt.plot(rounds_history, mae_history, 'r-o')
plt.xlabel('Round')
plt.ylabel('Mean Absolute Error (MAE)')
plt.title('Local Training MAE over Rounds')
plt.grid(True)
plt.tight_layout()
mae_plot_path = os.path.join(checkpoint_dir, 'mae_curve.png')
plt.savefig(mae_plot_path)
plt.close()

# 打印最终结果
print('\n====================== Local Training Results ======================')
print(f'Average MSE across all clients: {final_mse:.6f}')
print(f'Best MSE: {best_mse:.6f}')
print(f'Standard deviation: {final_std:.6f}')
print(f'MSE curve saved to: {mse_plot_path}')

# 打印每个客户端的MSE统计信息
print(f'\nMinimum client MSE: {np.min(final_mse_per_client):.6f}')
print(f'Maximum client MSE: {np.max(final_mse_per_client):.6f}')
print(f'Median client MSE: {np.median(final_mse_per_client):.6f}')

# 记录最终结果到日志
with open(log_file_path, 'a') as log_file:
    log_file.write(f"===== Training Completed =====\n")
    log_file.write(f"Final average MSE: {final_mse:.6f}\n")
    log_file.write(f"Best MSE: {best_mse:.6f}\n")
    log_file.write(f"Standard deviation: {final_std:.6f}\n")
    log_file.write(f"Minimum client MSE: {np.min(final_mse_per_client):.6f}\n")
    log_file.write(f"Maximum client MSE: {np.max(final_mse_per_client):.6f}\n")
    log_file.write(f"Median client MSE: {np.median(final_mse_per_client):.6f}\n")
    log_file.write(f"MSE curve saved to: {mse_plot_path}\n")
    log_file.write(f"Completion Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
