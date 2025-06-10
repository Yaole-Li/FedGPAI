import numpy as np
import argparse
import torch
import os
import time
import matplotlib.pyplot as plt
from torch import nn
from torch.optim import SGD
import copy
from pynvml import *
import psutil
from lib.datasets.data_loader import data_loader
from lib.FedGPAI.models import MLPFeatureExtractor, MLPRegressor

# 初始化NVML以监控GPU显存
nvmlInit()
handle = nvmlDeviceGetHandleByIndex(0)

def print_memory_usage(prefix=""):
    """打印当前内存和显存使用情况"""
    # 获取GPU显存信息
    info = nvmlDeviceGetMemoryInfo(handle)
    gpu_total = info.total / 1024**2  # MB
    gpu_used = info.used / 1024**2
    gpu_free = info.free / 1024**2
    
    # 获取系统内存信息
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    ram_used = mem_info.rss / 1024**2  # MB
    
    print(f"{prefix} | GPU: {gpu_used:.1f}/{gpu_total:.1f} MB (Free: {gpu_free:.1f} MB) | RAM: {ram_used:.1f} MB")

def parse_hidden_dims(dims_str):
    """
    解析字符串形式的隐藏层维度配置
    
    Args:
        dims_str: 以逗号分隔的隐藏层维度字符串，例如 "256,128,64"
        
    Returns:
        list: 隐藏层维度列表，例如 [256, 128, 64]
    """
    if not dims_str:
        return []
    return [int(dim) for dim in dims_str.split(',')]


class FedAvgModel(nn.Module):
    """用于 FedAvg 的 MLP 模型，包含特征提取器和回归器"""
    def __init__(self, input_dim, extractor_hidden_dims=None, regressor_hidden_dims=None, output_dim=32):
        super(FedAvgModel, self).__init__()
        if extractor_hidden_dims is None:
            extractor_hidden_dims = [256, 128, 64]
        if regressor_hidden_dims is None:
            regressor_hidden_dims = [32, 16]
        
        self.feature_extractor = MLPFeatureExtractor(input_dim, output_dim, hidden_dims=extractor_hidden_dims)
        self.regressor = MLPRegressor(output_dim, hidden_dims=regressor_hidden_dims)
    
    def forward(self, x):
        features = self.feature_extractor(x)
        return self.regressor(features)

def train(model, data_X, data_Y, optimizer, epochs=1):
    """训练模型一定轮次
    
    Args:
        model: 神经网络模型
        data_X: 输入特征
        data_Y: 目标标签
        optimizer: 优化器实例
        epochs: 本地训练轮次
        
    Returns:
        model: 训练后的模型
        loss: 最终损失值
    """
    model.train()
    criterion = nn.MSELoss()
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(data_X)
        loss = criterion(output, data_Y)
        loss.backward()
        optimizer.step()
    
    return model, loss.item()

def evaluate(model, data_X, data_Y):
    """评估模型
    
    Args:
        model: 神经网络模型
        data_X: 输入特征
        data_Y: 目标标签
        
    Returns:
        mse: 均方误差
        mae: 平均绝对误差
    """
    model.eval()
    with torch.no_grad():
        output = model(data_X)
        mse = torch.mean((output - data_Y) ** 2).item()
        mae = torch.mean(torch.abs(output - data_Y)).item()
    return mse, mae

def average_weights(w):
    """
    返回权重的平均值
    
    Args:
        w: 客户端模型权重列表
        
    Returns:
        平均后的权重
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg

def save_checkpoint(state, filename):
    """保存检查点"""
    torch.save(state, filename)
    print(f"检查点已保存到 {filename}")

def load_checkpoint(filename):
    """加载检查点"""
    return torch.load(filename)

# 解析命令行参数
parser = argparse.ArgumentParser()

# 数据集和任务相关参数
parser.add_argument("--dataset", default='Air', type=str, help="数据集名称")
parser.add_argument("--task", default='regression', type=str, help="任务类型")

# 客户端相关参数
parser.add_argument("--num_clients", default=400, type=int, help="客户端数量")
parser.add_argument("--client_ratio", default=1, type=float, help="每轮选择的客户端比例")
parser.add_argument("--num_samples", default=250, type=int, help="每个客户端的样本数量")
parser.add_argument("--test_ratio", default=0.2, type=float, help="测试集比例")

# 模型参数
parser.add_argument("--feature_dim", type=int, default=32, help="特征提取器输出维度")
parser.add_argument("--extractor_hidden_dims", type=str, default="256,128,64", help="特征提取器隐藏层维度，以逗号分隔")
parser.add_argument("--regressor_hidden_dims", type=str, default="32,16", help="回归器隐藏层维度，以逗号分隔")

# 训练参数
parser.add_argument("--lr", default=0.01, type=float, help="学习率")
parser.add_argument("--weight_decay", default=1e-5, type=float, help="权重衰减")
parser.add_argument("--global_rounds", default=20, type=int, help="全局联邦训练轮数")
parser.add_argument("--local_epochs", default=5, type=int, help="本地训练轮数")

# 检查点相关参数
parser.add_argument("--resume", action="store_true", help="是否从检查点继续训练")
parser.add_argument("--checkpoint", type=str, default="", help="检查点文件路径")

args = parser.parse_args()

# 加载数据集
print(f"正在加载 {args.dataset} 数据集...")
X, Y = data_loader(args)

# 获取数据维度和客户端数量
K = args.num_clients
input_dim = X[0].shape[1]  # 特征维度
print(f"数据维度: {input_dim}, 客户端数量: {K}")

# 设置设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 将隐藏层维度字符串转换为列表
extractor_hidden_dims = parse_hidden_dims(args.extractor_hidden_dims)
regressor_hidden_dims = parse_hidden_dims(args.regressor_hidden_dims)
print(f"特征提取器隐藏层: {extractor_hidden_dims}")
print(f"回归器隐藏层: {regressor_hidden_dims}")

# 初始化全局模型
global_model = FedAvgModel(
    input_dim=input_dim, 
    extractor_hidden_dims=extractor_hidden_dims,
    regressor_hidden_dims=regressor_hidden_dims,
    output_dim=args.feature_dim
).to(device)
global_weights = global_model.state_dict()

# 为每个客户端准备模型和数据
clients = {}
for k in range(K):
    # 创建客户端模型
    client_model = FedAvgModel(
        input_dim=input_dim, 
        extractor_hidden_dims=extractor_hidden_dims,
        regressor_hidden_dims=regressor_hidden_dims,
        output_dim=args.feature_dim
    ).to(device)
    
    clients[k] = {
        'model': client_model,
        'optimizer': SGD(client_model.parameters(), lr=args.lr, weight_decay=args.weight_decay),
        'X_train': torch.tensor(X[k][:int((1-args.test_ratio)*X[k].shape[0])], dtype=torch.float32).to(device),
        'Y_train': torch.tensor(Y[k][:int((1-args.test_ratio)*Y[k].shape[0])], dtype=torch.float32).to(device),
        'X_test': torch.tensor(X[k][int((1-args.test_ratio)*X[k].shape[0]):], dtype=torch.float32).to(device),
        'Y_test': torch.tensor(Y[k][int((1-args.test_ratio)*Y[k].shape[0]):], dtype=torch.float32).to(device)
    }
    # 设置初始权重为全局权重
    clients[k]['model'].load_state_dict(copy.deepcopy(global_weights))

# 创建检查点目录（加上时间戳）
current_time = time.strftime('%Y%m%d_%H%M%S')
checkpoint_dir = f"checkpoints/FedAvg_{args.dataset}_{args.num_clients}_{args.global_rounds}_{current_time}"
os.makedirs(checkpoint_dir, exist_ok=True)

# 设置日志文件
log_file = f"{checkpoint_dir}/FedAvg_{args.dataset}_{args.num_clients}_{args.global_rounds}.txt"
with open(log_file, 'w') as f:
    f.write(f"FedAvg on {args.dataset} Dataset\n")
    f.write(f"Clients: {args.num_clients}, Global Rounds: {args.global_rounds}, Local Epochs: {args.local_epochs}\n")
    f.write(f"Learning Rate: {args.lr}, Weight Decay: {args.weight_decay}\n")

print(f"检查点将保存到: {checkpoint_dir}")
print(f"日志将保存到: {log_file}")

# 跟踪训练和测试损失
train_loss_history = []
test_loss_history = []
test_mae_history = []

# 跟踪最小值
best_mse = float('inf')
best_mae = float('inf')

# 如果从检查点恢复
start_round = 0
if args.resume and args.checkpoint:
    checkpoint = load_checkpoint(args.checkpoint)
    global_model.load_state_dict(checkpoint['global_model_state_dict'])
    global_weights = global_model.state_dict()
    start_round = checkpoint['round'] + 1
    train_loss_history = checkpoint['train_loss_history']
    test_loss_history = checkpoint['test_loss_history']
    test_mae_history = checkpoint['test_mae_history']
    print(f"从轮次 {start_round} 恢复训练")

print(f"开始FedAvg训练 (共{args.global_rounds}轮)...")

for round in range(start_round, args.global_rounds):
    print(f"\n全局轮次 {round+1}/{args.global_rounds}")
    print_memory_usage("训练前")
    
    round_time_start = time.time()
    
    # 选择一部分客户端进行训练
    num_clients_to_select = max(1, int(K * args.client_ratio))
    client_indices = np.random.choice(range(K), num_clients_to_select, replace=False)
    print(f"选择了 {num_clients_to_select} 个客户端进行训练")
    
    # 收集本地更新
    local_weights = []
    local_losses = []
    
    # 客户端本地训练
    for idx in client_indices:
        # 获取客户端数据和模型
        local_model = clients[idx]['model']
        local_optimizer = clients[idx]['optimizer']
        local_X = clients[idx]['X_train']
        local_Y = clients[idx]['Y_train']
        
        # 更新本地模型为全局权重
        local_model.load_state_dict(copy.deepcopy(global_weights))
        local_optimizer = SGD(local_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        
        # 本地训练
        local_model, loss = train(
            model=local_model,
            data_X=local_X,
            data_Y=local_Y,
            optimizer=local_optimizer,
            epochs=args.local_epochs
        )
        
        # 保存本地权重和损失
        local_weights.append(copy.deepcopy(local_model.state_dict()))
        local_losses.append(loss)
    
    # 聚合本地模型权重获得新的全局模型
    global_weights = average_weights(local_weights)
    global_model.load_state_dict(global_weights)
    
    # 计算平均训练损失
    train_loss = sum(local_losses) / len(local_losses)
    train_loss_history.append(train_loss)
    
    # 在测试集上评估全局模型
    test_mse_sum = 0
    test_mae_sum = 0
    for k in range(K):
        mse, mae = evaluate(
            model=global_model,
            data_X=clients[k]['X_test'],
            data_Y=clients[k]['Y_test']
        )
        test_mse_sum += mse
        test_mae_sum += mae
    
    test_mse = test_mse_sum / K
    test_mae = test_mae_sum / K
    test_loss_history.append(test_mse)
    test_mae_history.append(test_mae)
    
    # 更新最小值
    best_mse = min(best_mse, test_mse)
    best_mae = min(best_mae, test_mae)
    
    # 计算轮次时间
    round_time = time.time() - round_time_start
    
    # 打印结果
    print(f"  训练损失: {train_loss:.6f}")
    print(f"  测试MSE: {test_mse:.6f}")
    print(f"  测试MAE: {test_mae:.6f}")
    print(f"  轮次用时: {round_time:.2f} 秒")
    
    # 记录到日志文件
    with open(log_file, 'a') as f:
        f.write(f"轮次 {round+1}\n")
        f.write(f"  训练损失: {train_loss:.6f}\n")
        f.write(f"  测试MSE: {test_mse:.6f}\n")  
        f.write(f"  测试MAE: {test_mae:.6f}\n")
        f.write(f"  轮次用时: {round_time:.2f} 秒\n\n")
    
    # 每5轮保存检查点或者最后一轮
    if (round+1) % 5 == 0 or round == args.global_rounds - 1:
        checkpoint_path = f"{checkpoint_dir}/round_{round+1}.pt"
        save_checkpoint({
            'round': round,
            'global_model_state_dict': global_model.state_dict(),
            'train_loss_history': train_loss_history,
            'test_loss_history': test_loss_history,
            'test_mae_history': test_mae_history
        }, checkpoint_path)

# 保存最终结果
print("\n训练完成，保存结果...")

# 绘制MSE曲线
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(test_loss_history) + 1), test_loss_history, 'b-o')
plt.xlabel('Round')
plt.ylabel('Mean Squared Error (MSE)')
plt.title('FedAvg Training MSE over Rounds')
plt.grid(True)
plt.tight_layout()
mse_plot_path = os.path.join(checkpoint_dir, 'mse_curve.png')
plt.savefig(mse_plot_path)
plt.close()

# 绘制MAE曲线
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(test_mae_history) + 1), test_mae_history, 'r-o')
plt.xlabel('Round')
plt.ylabel('Mean Absolute Error (MAE)')
plt.title('FedAvg Training MAE over Rounds')
plt.grid(True)
plt.tight_layout()
mae_plot_path = os.path.join(checkpoint_dir, 'mae_curve.png')
plt.savefig(mae_plot_path)
plt.close()

print(f"MSE curve saved to {mse_plot_path}")
print(f"MAE curve saved to {mae_plot_path}")

# 打印最终结果
final_test_mse = test_loss_history[-1]
final_test_mae = test_mae_history[-1]
print(f"FedAvg final test MSE: {final_test_mse:.6f}")
print(f"FedAvg final test MAE: {final_test_mae:.6f}")
print(f"FedAvg best MSE: {best_mse:.6f}")
print(f"FedAvg best MAE: {best_mae:.6f}")

# 将最终结果写入日志
with open(log_file, 'a') as f:
    f.write("\n===== Training Completed =====\n")
    f.write(f"Final test MSE: {final_test_mse:.6f}\n")
    f.write(f"Final test MAE: {final_test_mae:.6f}\n")
    f.write(f"Best MSE: {best_mse:.6f}\n")
    f.write(f"Best MAE: {best_mae:.6f}\n")
    f.write(f"MSE curve saved to: {mse_plot_path}\n")
    f.write(f"MAE curve saved to: {mae_plot_path}\n")
    f.write(f"Completion Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

print("\n训练结束!")
