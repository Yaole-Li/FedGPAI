import numpy as np
import argparse
import torch
import time
import os
import gc
import matplotlib.pyplot as plt
from lib.FedGPAI.models import MLPFeatureExtractor, MLPRegressor
from lib.datasets.data_loader import data_loader
from lib.FedGPAI.FedGPAI_advanced import FedGPAI_advanced

# 命令行参数设置
parser = argparse.ArgumentParser()

# 数据集和任务相关参数
parser.add_argument("--dataset", default='Air', type=str, help="数据集名称")
parser.add_argument("--task", default='regression', type=str, help="任务类型")

# 客户端相关参数
parser.add_argument("--num_clients", default=400, type=int, help="客户端数量")
parser.add_argument("--num_samples", default=250, type=int, help="每个客户端的样本数量")
parser.add_argument("--test_ratio", default=0.2, type=float, help="测试集比例")

# 模型相关参数
parser.add_argument("--regularizer", default=1e-6, type=float, help="正则化参数")
parser.add_argument("--global_rounds", default=20, type=int, help="全局联邦训练轮数")
parser.add_argument("--local_rounds", default=5, type=int, help="本地训练轮数")
parser.add_argument("--train_head_epochs", default=5, type=int, help="训练回归器的轮数")
parser.add_argument("--extractor_hidden_dims", default="256,128,64", type=str, help="特征提取器MLP隐藏层维度，以逗号分隔的字符串")
parser.add_argument("--regressor_hidden_dims", default="32,16", type=str, help="回归器MLP隐藏层维度，以逗号分隔的字符串")
parser.add_argument("--output_dim", default=32, type=int, help="特征提取器输出维度")

# 检查点相关参数
parser.add_argument("--resume", action="store_true", help="是否从检查点继续训练")
parser.add_argument("--checkpoint", type=str, default="", help="检查点文件路径")

args = parser.parse_args()

# 设置学习率
args.eta = 1/np.sqrt(args.num_samples)

# 设置设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 加载数据集
print(f"正在加载 {args.dataset} 数据集...")
X, Y = data_loader(args)

# 获取数据维度
M, N = X[0].shape
K = args.num_clients
M *= K

# 初始化模型配置

# 创建保存模型的目录
# 使用方法名称_数据集_客户端数量_全局联邦训练轮数_时间戳作为文件夹名称
current_time = time.strftime('%Y%m%d_%H%M%S')
checkpoint_dir = os.path.join("checkpoints", f"FedRep_{args.dataset}_{args.num_clients}_{args.global_rounds}_{current_time}")
os.makedirs(checkpoint_dir, exist_ok=True)
print(f"检查点将保存到: {checkpoint_dir}")

# 创建日志文件
log_file_name = f"FedRep_{args.dataset}_{args.num_clients}_{args.global_rounds}.txt"
log_file_path = os.path.join(checkpoint_dir, log_file_name)

# 记录训练起始信息到日志
with open(log_file_path, 'w') as log_file:
    log_file.write(f"===== 训练开始 =====\n")
    log_file.write(f"方法: FedRep\n")
    log_file.write(f"数据集: {args.dataset}\n")
    log_file.write(f"客户端数量: {args.num_clients}\n")
    log_file.write(f"全局训练轮数: {args.global_rounds}\n")
    log_file.write(f"时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

print(f"日志将保存到: {log_file_path}")

# 创建保存模型的目录
os.makedirs(checkpoint_dir, exist_ok=True)

# 初始化变量
start_epoch = 0

# 初始化性能评估指标
mse = torch.zeros((args.num_samples, 1), dtype=torch.float32).to(device)
m = torch.zeros((K, args.num_samples, args.global_rounds), dtype=torch.float32).to(device)

# 跟踪训练过程中的MSE和MAE
mse_history = []
mae_history = []
rounds_history = []


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

# 跟踪最小值
best_mse = float('inf')
best_mae = float('inf')


def get_fedrep_models(input_dim, args):
    """
    获取FedRep模型实例
    
    Args:
        input_dim: 输入维度
        args: 参数对象
        
    Returns:
        local_models: 本地模型列表
        global_model: 全局模型
    """
    # 解析隐藏层维度
    extractor_hidden_dims = parse_hidden_dims(args.extractor_hidden_dims)
    regressor_hidden_dims = parse_hidden_dims(args.regressor_hidden_dims)
    
    print(f"\n创建MLP模型 - 输入维度: {input_dim}")
    print(f"  特征提取器: 隐藏层{extractor_hidden_dims}, 输出维度{args.output_dim}")
    print(f"  回归器: 隐藏层{regressor_hidden_dims}, 输出维度 1")
    
    # 创建全局模型（只包含回归器部分）
    global_model = FedGPAI_advanced(
        lam=args.regularizer, 
        rf_feature=input_dim,  # 这里直接使用输入维度
        eta=args.eta, 
        regressor_type='mlp', 
        extractor_hidden_dims=extractor_hidden_dims,
        regressor_hidden_dims=regressor_hidden_dims,
        output_dim=args.output_dim,
        num_clients=args.num_clients, 
        is_global=True
    )
    
    # 创建每个客户端的本地模型
    local_models = []
    
    for i in range(args.num_clients):
        # 创建本地模型
        local_model = FedGPAI_advanced(
            lam=args.regularizer, 
            rf_feature=input_dim,  # 这里直接使用输入维度
            eta=args.eta, 
            regressor_type='mlp', 
            extractor_hidden_dims=extractor_hidden_dims,
            regressor_hidden_dims=regressor_hidden_dims,
            output_dim=args.output_dim,
            num_clients=args.num_clients, 
            is_global=False
        )
        
        # FedGPAI_advanced 已经在初始化时创建了特征提取器和回归器
        # 不需要再手动设置
        
        local_models.append(local_model)
    
    return local_models, global_model


# 从检查点恢复训练
if args.resume and args.checkpoint:
    if os.path.exists(args.checkpoint):
        print(f"加载检查点: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint)
        start_epoch = checkpoint['epoch']
        # TODO: 加载其他状态
        
        # 如果有全局模型状态，则需要在创建模型时加载
        global_model_state = checkpoint.get('global_model', None)
        last_mse = checkpoint.get('mse', 0)
        last_mae = checkpoint.get('mae', 0)
        
        print(f"恢复训练从轮次: {start_epoch}")
        print(f"上一轮次MSE: {last_mse:.6f}, MAE: {last_mae:.6f}")
    else:
        print(f"检查点文件不存在: {args.checkpoint}, 从头开始训练")

print(f"开始FedRep联邦学习训练 ({args.global_rounds} 轮全局训练, {args.local_rounds} 轮本地训练)...")

# 执行联邦学习训练过程
for cc in range(start_epoch, args.global_rounds):
    print(f"\n全局轮次 {cc+1}/{args.global_rounds}")
    
    if cc == 0:
        # 获取输入维度
        input_dim = X[0].shape[1]
        
        # 获取FedRep模型
        local_models, global_model = get_fedrep_models(input_dim, args)
    
    # 如果是从检查点恢复训练的第一个训练轮次，加载模型状态
    if cc == start_epoch and args.resume and args.checkpoint and 'global_model_state' in locals():
        if global_model_state is not None and hasattr(global_model, 'load_state_dict'):
            try:
                global_model.load_state_dict(global_model_state)
                print(f"  成功加载全局回归器状态")
            except Exception as e:
                print(f"  加载模型状态失败: {str(e)}")
        else:
            print("  没有可用的模型状态或模型不支持加载状态")
    
    # 创建误差记录器
    e = torch.zeros((args.num_samples, K), dtype=torch.float32).to(device)
    
    # 训练过程
    for i in range(args.num_samples):
        if i % 50 == 0:
            print(f"  评估进度: 样本 {i}/{args.num_samples}")
            
        all_regressors = []  # 收集所有客户端的回归器
        
        # 客户端本地训练
        for j in range(K):
            # 显示客户端本地训练信息
            if i % 50 == 0 and j == 0:
                print(f"    客户端{j+1}开始本地训练: 全局轮次 {cc+1}, 样本 {i}")
                print(f"    将执行 {args.local_rounds} 轮本地训练 (前{args.train_head_epochs}轮训练回归器, 后{args.local_rounds-args.train_head_epochs}轮训练特征提取器)")
            
            # 将全局回归器复制到本地模型，保持特征提取器不变
            local_model = local_models[j]
            
            # 深复制全局模型的回归器到本地模型
            for target_param, source_param in zip(local_model.regressor.parameters(), global_model.regressor.parameters()):
                target_param.data.copy_(source_param.data)
            
            # 将数据转换为PyTorch张量
            x_j = torch.tensor(X[j][i:i+1, :], dtype=torch.float32).to(device)
            y_j = torch.tensor(Y[j][i], dtype=torch.float32).to(device)
            
            # FedRep本地训练（分两个阶段）
            for local_round in range(args.local_rounds):
                # 取消所有梯度 
                if hasattr(local_model.feature_extractor, 'zero_grad'):
                    local_model.feature_extractor.zero_grad()
                if hasattr(local_model.regressor, 'zero_grad'):
                    local_model.regressor.zero_grad()
                
                # 前向传播
                x_features = local_model.feature_extractor(x_j)  
                outputs = local_model.regressor(x_features)  
                
                # 计算损失
                loss = torch.mean((outputs - y_j)**2)
                
                # 反向传播
                loss.backward()
                
                # 手动更新参数(模拟优化器)
                with torch.no_grad():
                    if local_round < args.train_head_epochs:
                        # 阶段1：只更新回归器参数
                        for param in local_model.regressor.parameters():
                            if param.grad is not None:
                                param.data.sub_(args.eta * param.grad.data)
                    else:
                        # 阶段2：只更新特征提取器参数
                        for param in local_model.feature_extractor.parameters():
                            if param.grad is not None:
                                param.data.sub_(args.eta * param.grad.data)
            
            # 训练结束后，使用本地模型进行预测
            with torch.no_grad():
                x_features = local_model.feature_extractor(x_j)
                outputs = local_model.regressor(x_features)
            
            # 记录当前轮次的均方误差
            current_mse = (outputs - y_j)**2
            m[j, i, cc] = current_mse
            
            # 计算累积误差
            if i == 0:
                e[i, j] = current_mse
            else:
                e[i, j] = (1/(i+1)) * ((i*e[i-1, j]) + current_mse)
            
            # 收集回归器参数用于全局聚合
            from copy import deepcopy
            all_regressors.append(deepcopy(local_model.regressor))
        
        # 全局模型聚合（只聚合回归器）
        if all_regressors:
            # MLP模型参数聚合
            # 初始化参数字典来存储聚合后的参数
            global_state_dict = {}
            
            # 遍历全局模型的每个参数
            for name, param in global_model.regressor.named_parameters():
                # 初始化等价参数的零张量
                global_state_dict[name] = torch.zeros_like(param.data)
                
                # 加和各客户端模型的参数
                for model in all_regressors:
                    for client_name, client_param in model.named_parameters():
                        if client_name == name:
                            global_state_dict[name] += client_param.data
                            
                # 取平均值
                global_state_dict[name] /= len(all_regressors)
            
            # 更新全局模型参数
            for name, param in global_model.regressor.named_parameters():
                param.data.copy_(global_state_dict[name])
                
            # print(f"  已聚合 {len(all_regressors)} 个客户端的MLP回归器参数")
    
    # 计算平均误差
    mse = (1/(cc+1)) * ((cc*mse)+torch.reshape(torch.mean(e, dim=1), (-1, 1)))
    
    # 每轮记录MSE和MAE
    current_mse = mse[-1].item()
    current_mae = torch.mean(torch.sqrt(mse[-1])).item()
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
        
        # 构建检查点文件名和路径
        checkpoint_filename = f"epoch_{cc+1}.pt"
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_filename)
        
        # 保存模型和训练状态
        checkpoint = {
            'epoch': cc + 1,
            'global_model': global_model.state_dict() if hasattr(global_model, 'state_dict') else None,
            'local_models': [model.state_dict() if hasattr(model, 'state_dict') else None for model in local_models],
            'mse': mse[-1].item(),
            'mae': current_mae
        }
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
plt.title('FedRep Training MSE over Rounds')
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
plt.title('FedRep Training MAE over Rounds')
plt.grid(True)
plt.tight_layout()
mae_plot_path = os.path.join(checkpoint_dir, 'mae_curve.png')
plt.savefig(mae_plot_path)
plt.close()

# 打印最终结果
final_mse = mse[-1].item()
final_mae = torch.mean(torch.sqrt(mse[-1])).item()
final_std = torch.std(torch.mean(torch.mean(m, dim=2), dim=1)).item()
print(f'FedRep final MSE: {final_mse:.6f}')
print(f'FedRep final MAE: {final_mae:.6f}')
print(f'FedRep best MSE: {best_mse:.6f}')
print(f'FedRep best MAE: {best_mae:.6f}')
print(f'FedRep standard deviation: {final_std:.6f}')
print(f'MSE curve saved to: {mse_plot_path}')
print(f'MAE curve saved to: {mae_plot_path}')

# 记录最终结果到日志
with open(log_file_path, 'a') as log_file:
    log_file.write(f"===== Training Completed =====\n")
    log_file.write(f"Final MSE: {final_mse:.6f}\n")
    log_file.write(f"Final MAE: {final_mae:.6f}\n")
    log_file.write(f"Best MSE: {best_mse:.6f}\n")
    log_file.write(f"Best MAE: {best_mae:.6f}\n")
    log_file.write(f"Standard Deviation: {final_std:.6f}\n")
    log_file.write(f"MSE curve saved to: {mse_plot_path}\n")
    log_file.write(f"MAE curve saved to: {mae_plot_path}\n")
    log_file.write(f"Completion Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
