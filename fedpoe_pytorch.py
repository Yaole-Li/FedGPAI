import argparse
import torch
import numpy as np
import os
import gc
import time
import matplotlib.pyplot as plt
from lib.datasets.data_loader import data_loader
from lib.FedPOE.get_FedPOE_pytorch import get_FedPOE

parser = argparse.ArgumentParser()

# 数据集和任务相关参数
parser.add_argument("--dataset", default='WEC', type=str, help="数据集名称")
parser.add_argument("--task", default='regression', type=str, help="任务类型")

# 客户端相关参数
parser.add_argument("--num_clients", default=400, type=int, help="客户端数量")
parser.add_argument("--num_samples", default=250, type=int, help="每个客户端的样本数量")
parser.add_argument("--test_ratio", default=0.2, type=float, help="测试集比例")

# 模型相关参数
parser.add_argument("--num_random_features", default=100, type=int, help="随机特征数量")
parser.add_argument("--regularizer", default=1e-6, type=float, help="正则化参数")
parser.add_argument("--global_rounds", default=20, type=int, help="全局训练轮数")

args = parser.parse_args()

# 设置学习率
args.eta = 1/np.sqrt(args.num_samples)

# 加载数据集
print(f"正在加载 {args.dataset} 数据集...")
X, Y = data_loader(args)

# 获取数据维度
M, N = X[0].shape
K = args.num_clients
M *= K

# 设置随机核参数
print("初始化随机特征...")
gamma = []
num_rbf = 3
for i in range(num_rbf):
    gamma.append(10**(i-1))
gamma = np.array(gamma)

# 设置随机特征数量
n_components = args.num_random_features

# 初始化权重
w = torch.ones((K, np.prod(gamma.shape)), dtype=torch.float32)
w_loc = torch.ones((K, np.prod(gamma.shape)), dtype=torch.float32)
a = torch.ones((K, 1), dtype=torch.float32)
b = torch.ones((K, 1), dtype=torch.float32)

# 初始化设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 创建保存模型的目录（加上时间戳）
current_time = time.strftime('%Y%m%d_%H%M%S')
checkpoint_dir = os.path.join("checkpoints", f"FedPOE_{args.dataset}_{args.num_clients}_{args.global_rounds}_{current_time}")
os.makedirs(checkpoint_dir, exist_ok=True)
print(f"检查点将保存到: {checkpoint_dir}")

# 创建日志文件
log_file_name = f"FedPOE_{args.dataset}_{args.num_clients}_{args.global_rounds}_{current_time}.txt"
log_file_path = os.path.join(checkpoint_dir, log_file_name)

# 记录训练起始信息到日志
with open(log_file_path, 'w') as log_file:
    log_file.write(f"===== 训练开始 =====\n")
    log_file.write(f"方法: FedPOE\n")
    log_file.write(f"数据集: {args.dataset}\n")
    log_file.write(f"客户端数量: {args.num_clients}\n")
    log_file.write(f"全局训练轮数: {args.global_rounds}\n")
    log_file.write(f"时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

print(f"日志将保存到: {log_file_path}")

w = w.to(device)
w_loc = w_loc.to(device)
a = a.to(device)
b = b.to(device)

# 初始化评估指标
mse = torch.zeros((args.num_samples, 1), dtype=torch.float32).to(device)
m = torch.zeros((K, args.num_samples, args.global_rounds), dtype=torch.float32).to(device)

# 跟踪训练过程中的MSE和MAE
mse_history = []
mae_history = []
rounds_history = []

print(f"开始FedPOE训练 (共{args.global_rounds}轮)...")
for cc in range(args.global_rounds):
    print(f"\n全局轮次 {cc+1}/{args.global_rounds}")
    
    # 生成随机特征
    ran_feature = torch.zeros((N, n_components, gamma.shape[0]), dtype=torch.float32)
    for i in range(num_rbf):
        ran_feature[:, :, i] = torch.randn(N, n_components) * torch.sqrt(torch.tensor(1/gamma[i], dtype=torch.float32))
    
    ran_feature = ran_feature.to(device)
    
    # 获取FedPOE模型
    alg_loc, alg = get_FedPOE(ran_feature, args)
    
    # 初始化误差矩阵
    e = torch.zeros((args.num_samples, K), dtype=torch.float32).to(device)
    
    # 训练过程
    for i in range(args.num_samples):
        if i % 50 == 0:
            print(f"  训练进度: 样本 {i}/{args.num_samples}")
            
        agg_grad = []
        
        for j in range(K):
            # 将NumPy数组转换为PyTorch张量
            x_j = torch.tensor(X[j][i:i+1, :], dtype=torch.float32).to(device)
            y_j = torch.tensor(Y[j][i], dtype=torch.float32).to(device)
            
            # 全局模型预测
            f_RF_fed, f_RF_p, X_features = alg.predict(x_j, w[j:j+1, :])
            
            # 全局模型权重更新
            w[j:j+1, :], local_grad = alg.local_update(f_RF_p, y_j, w[j:j+1, :], X_features)
            
            # 本地模型预测
            f_RF_loc, f_RF_p_loc, X_features_loc = alg_loc[j].predict(x_j, w_loc[j:j+1, :])
            
            # 本地模型权重更新
            w_loc[j:j+1, :], local_grad_loc = alg_loc[j].local_update(f_RF_p_loc, y_j, w_loc[j:j+1, :], X_features_loc)
            
            # 组合预测
            f_RF = (a[j, 0] * f_RF_fed + b[j, 0] * f_RF_loc) / (a[j, 0] + b[j, 0])
            
            # 计算损失
            l_fed = (f_RF_fed - y_j)**2
            l_loc = (f_RF_loc - y_j)**2
            
            # 计算指数项
            exp_term_fed = torch.exp(-args.eta * l_fed)
            exp_term_loc = torch.exp(-args.eta * l_loc)
            
            # 直接更新a和b的值，避免广播问题
            # 将两个标量值转换为浮点数
            a[j, 0] = a[j, 0] * float(exp_term_fed)
            b[j, 0] = b[j, 0] * float(exp_term_loc)
            
            # 本地模型更新
            alg_loc[j].global_update([local_grad_loc])
            
            # 收集梯度用于全局更新
            agg_grad.append(local_grad)
            
            # 记录当前轮次的均方误差
            m[j, i, cc] = (f_RF - y_j)**2
            
            # 计算累积误差
            if i == 0:
                e[i, j] = (f_RF - y_j)**2
            else:
                e[i, j] = (1 / (i + 1)) * ((i * e[i-1, j]) + ((f_RF - y_j)**2))
        
        # 全局模型更新
        alg.global_update(agg_grad)
    
    # 更新总体MSE
    mse = (1 / (cc + 1)) * ((cc * mse) + torch.reshape(torch.mean(e, dim=1), (-1, 1)))
    
    # 每轮记录MSE和MAE
    current_mse = mse[-1].item()
    current_mae = torch.mean(torch.sqrt(mse[-1])).item()
    mse_history.append(current_mse)
    mae_history.append(current_mae)
    rounds_history.append(cc+1)
    
    # 每5轮计算并输出一次MAE
    if (cc+1) % 5 == 0 or cc == 0:
        print(f"\n  Round {cc+1} - MSE: {current_mse:.6f}, MAE: {current_mae:.6f}")
        
        # 将结果写入日志文件
        with open(log_file_path, 'a') as log_file:
            log_file.write(f"轮次 {cc+1}/{args.global_rounds}\n")
            log_file.write(f"MSE: {current_mse:.6f}\n")
            log_file.write(f"MAE: {current_mae:.6f}\n")
            log_file.write(f"时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # 保存模型
        checkpoint = {
            'epoch': cc + 1,
            'global_model': alg.state_dict() if hasattr(alg, 'state_dict') else None,
            'w': w,
            'w_loc': w_loc,
            'a': a,
            'b': b,
            'mse': mse[-1].item(),
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

# 绘制和保存MSE曲线
plt.figure(figsize=(10, 6))
plt.plot(rounds_history, mse_history, 'b-o')
plt.xlabel('Round')
plt.ylabel('Mean Squared Error (MSE)')
plt.title('FedPOE Training MSE over Rounds')
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
plt.title('FedPOE Training MAE over Rounds')
plt.grid(True)
plt.tight_layout()
mae_plot_path = os.path.join(checkpoint_dir, 'mae_curve.png')
plt.savefig(mae_plot_path)
plt.close()

# 打印最终结果
final_mse = mse[-1].item()
final_std = torch.std(torch.mean(torch.mean(m, dim=2), dim=1)).item()
print(f'FedPOE final MSE: {final_mse:.6f}')
print(f'FedPOE standard deviation: {final_std:.6f}')
print(f'MSE curve saved to: {mse_plot_path}')
print(f'MAE curve saved to: {mae_plot_path}')

# 记录最终结果到日志
with open(log_file_path, 'a') as log_file:
    log_file.write(f"===== Training Completed =====\n")
    log_file.write(f"Final MSE: {final_mse:.6f}\n")
    log_file.write(f"Standard Deviation: {final_std:.6f}\n")
    log_file.write(f"MSE curve saved to: {mse_plot_path}\n")
    log_file.write(f"MAE curve saved to: {mae_plot_path}\n")
    log_file.write(f"Completion Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
