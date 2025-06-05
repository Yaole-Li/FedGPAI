import numpy as np
import argparse
import torch
import os
import gc
from lib.datasets.data_loader import data_loader
from lib.FedGPAI.get_FedGPAI import get_FedGPAI

parser = argparse.ArgumentParser()

# 数据集和任务相关参数
parser.add_argument("--dataset", default='Air', type=str, help="数据集名称")
parser.add_argument("--task", default='regression', type=str, help="任务类型")

# 客户端相关参数
parser.add_argument("--num_clients", default=100, type=int, help="客户端数量")
parser.add_argument("--num_samples", default=250, type=int, help="每个客户端的样本数量")
parser.add_argument("--test_ratio", default=0.2, type=float, help="测试集比例")

# 模型相关参数
parser.add_argument("--num_random_features", default=100, type=int, help="随机特征数量")
parser.add_argument("--regularizer", default=1e-6, type=float, help="正则化参数")
parser.add_argument("--global_rounds", default=50, type=int, help="训练轮数")
parser.add_argument("--local_rounds", default=5, type=int, help="本地训练轮数")

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

# 初始化权重 - 每个客户端都有自己的独立权重
w_local = [torch.ones((1, np.prod(gamma.shape)), dtype=torch.float32) for _ in range(K)]

# 初始化设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 将所有张量移到相应设备
w_local = [w.to(device) for w in w_local]

# 创建保存模型的目录
os.makedirs("checkpoints", exist_ok=True)

# 初始化性能评估指标
mse_local = [torch.zeros((args.num_samples, 1), dtype=torch.float32).to(device) for _ in range(K)]
m_local = [torch.zeros((args.num_samples, args.global_rounds), dtype=torch.float32).to(device) for _ in range(K)]

print(f"开始本地训练 (每个客户端独立训练 {args.global_rounds} 轮)...")

# 执行本地训练过程 - 每个客户端完全独立
for cc in range(args.global_rounds):
    print(f"\n轮次 {cc+1}/{args.global_rounds}")
    
    # 生成随机特征
    ran_feature = torch.zeros((N, n_components, gamma.shape[0]), dtype=torch.float32)
    for i in range(num_rbf):
        ran_feature[:, :, i] = torch.randn(N, n_components) * torch.sqrt(torch.tensor(1/gamma[i], dtype=torch.float32))
        
    # 移动到相应设备
    ran_feature = ran_feature.to(device)
    
    # 为每个客户端单独获取模型
    alg_local = []
    for j in range(K):
        # 由于本地版本不需要全局模型，我们只初始化本地模型
        local_models, _, _ = get_FedGPAI(ran_feature, args)
        alg_local.append(local_models[j])
    
    # 创建误差记录器
    e_local = [torch.zeros((args.num_samples, 1), dtype=torch.float32).to(device) for _ in range(K)]
    
    # 每个客户端独立训练
    for j in range(K):
        if j % 10 == 0:
            print(f"  训练客户端 {j+1}/{K}")
        
        # 样本级训练
        for i in range(args.num_samples):
            # 将NumPy数组转换为PyTorch张量并移动到指定设备
            x_j = torch.tensor(X[j][i:i+1, :], dtype=torch.float32).to(device)
            y_j = torch.tensor(Y[j][i], dtype=torch.float32).to(device)
            
            # 本地模型预测
            f_RF_loc, f_RF_p, X_features = alg_local[j].predict(x_j, w_local[j])
            
            # 本地模型权重更新 - 没有联邦学习，只有本地更新
            w_local[j], local_grad = alg_local[j].local_update(f_RF_p, y_j, w_local[j], X_features)
            
            # 记录误差
            m_local[j][i, cc] = (f_RF_loc - y_j)**2
            
            # 计算累积误差
            if i == 0:
                e_local[j][i, 0] = (f_RF_loc - y_j)**2
            else:
                e_local[j][i, 0] = (1/(i+1)) * ((i*e_local[j][i-1, 0]) + ((f_RF_loc - y_j)**2))
    
    # 计算平均误差
    for j in range(K):
        mse_local[j] = (1/(cc+1)) * ((cc*mse_local[j]) + e_local[j])
    
    # 计算所有客户端的平均MSE，用于显示
    avg_mse = torch.mean(torch.stack([mse[args.num_samples-1, 0] for mse in mse_local]))
    
    # 每5轮计算并输出一次MAE
    if (cc+1) % 5 == 0 or cc == 0:
        current_mae = torch.mean(torch.sqrt(avg_mse)).item()
        print(f"\n  当前轮次 {cc+1} 的平均MAE为: {current_mae:.6f}")
        
        # 保存模型
        checkpoint = {
            'epoch': cc + 1,
            'local_models': [model.state_dict() if hasattr(model, 'state_dict') else None for model in alg_local],
            'w_local': w_local,
            'mse': avg_mse.item(),
            'mae': current_mae
        }
        torch.save(checkpoint, f"checkpoints/local_checkpoint_epoch_{cc+1}.pt")
        print(f"  模型已保存到: checkpoints/local_checkpoint_epoch_{cc+1}.pt")
    
    # 清理显存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

# 计算每个客户端的最终MSE
final_mse_per_client = [mse[args.num_samples-1, 0].item() for mse in mse_local]
final_mse = np.mean(final_mse_per_client)
final_std = np.std(final_mse_per_client)

# 打印最终结果
print('\n====================== 本地训练结果 ======================')
print(f'所有客户端平均MSE: {final_mse:.6f}')
print(f'所有客户端标准差: {final_std:.6f}')

# 打印每个客户端的MSE统计信息
print(f'\n客户端MSE最小值: {np.min(final_mse_per_client):.6f}')
print(f'客户端MSE最大值: {np.max(final_mse_per_client):.6f}')
print(f'客户端MSE中位数: {np.median(final_mse_per_client):.6f}')
