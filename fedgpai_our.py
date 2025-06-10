import numpy as np
import argparse
import torch
import os
import gc
import psutil
import time
import matplotlib.pyplot as plt
from datetime import datetime
from pynvml import *
from lib.datasets.data_loader import data_loader
from lib.FedGPAI.get_FedGPAI import get_FedGPAI
from lib.FedGPAI.get_FedGPAI_advanced import get_FedGPAI_advanced
from lib.FedGPAI.FedGPAI_regression import FedGPAI_regression
from lib.FedGPAI.FedGPAI_advanced import FedGPAI_advanced

# 初始化NVML以监控GPU显存
nvmlInit()
handle = nvmlDeviceGetHandleByIndex(0)  # 假设使用第一个GPU

def get_lr_decay_factor(epoch, args):
    """计算学习率衰减因子
    
    Args:
        epoch: 当前训练轮数
        args: 命令行参数
        
    Returns:
        学习率衰减因子，范围[args.lr_min_factor, 1.0]
    """
    # 如果没启用衰减或者还未到开始衰减轮数，返回1.0（无衰减）
    if not args.lr_decay or epoch < args.lr_decay_start:
        return 1.0
    
    # 计算衰减当前实际已衰减轮数
    effective_epoch = epoch - args.lr_decay_start
    
    if effective_epoch <= 0:
        return 1.0
    
    if args.lr_decay_type == "exponential":
        # 指数衰减: 每轮按照特定比例衰减，缓慢版本
        decay = args.lr_decay_rate ** effective_epoch
    
    elif args.lr_decay_type == "step":
        # 步进式衰减: 每固定步长后衰减
        decay = args.lr_decay_rate ** (effective_epoch // args.lr_step_size)
    
    elif args.lr_decay_type == "cosine":
        # 余弦衰减：最平滑的衰减方式
        remaining_epochs = args.global_rounds - args.lr_decay_start
        decay = 0.5 * (1 + np.cos(np.pi * effective_epoch / remaining_epochs))
    
    else:  # 默认使用指数衰减
        decay = args.lr_decay_rate ** effective_epoch
    
    # 限制最小学习率
    return max(args.lr_min_factor, decay)

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

def track_tensors():
    """跟踪当前所有张量的数量和大小"""
    total_size = 0
    total_num = 0
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                obj_size = obj.nelement() * obj.element_size() / 1024**2  # MB
                total_size += obj_size
                total_num += 1
        except:
            pass
    return total_num, total_size

parser = argparse.ArgumentParser()

# 数据集和任务相关参数
parser.add_argument("--dataset", default='Air', type=str, help="数据集名称")
parser.add_argument("--task", default='regression', type=str, help="任务类型")

# 客户端相关参数
parser.add_argument("--num_clients", default=400, type=int, help="客户端数量")
parser.add_argument("--num_samples", default=300, type=int, help="每个客户端的样本数量")
parser.add_argument("--test_ratio", default=0.2, type=float, help="测试集比例")

# 模型相关参数
parser.add_argument("--hidden_dim", default=64, type=int, help="隐藏层维度")
parser.add_argument("--max_lr", default=0.1, type=float, help="学习率上限，默认为0.01")
parser.add_argument("--use_fixed_lr", default=False, type=bool, help="是否使用固定学习率而非自动计算学习率")
parser.add_argument("--fixed_lr", default=0.005, type=float, help="如果使用固定学习率，该值将被使用")
parser.add_argument("--num_random_features", default=100, type=int, help="随机特征数量")
parser.add_argument("--regularizer", default=1e-6, type=float, help="正则化参数")
parser.add_argument("--global_rounds", default=20, type=int, help="全局联邦训练轮数")
parser.add_argument("--local_rounds", default=5, type=int, help="本地训练轮数")

# 回归器相关参数
parser.add_argument("--regressor_type", default='mlp', type=str, choices=['linear', 'mlp'], help="回归器类型: linear或mlp")
parser.add_argument("--hidden_dims", default=[32, 16], type=int, nargs='+', help="MLP回归器的隐藏层维度列表")

# 特征提取器相关参数
parser.add_argument("--extractor_hidden_dims", default=[256, 128, 64], type=int, nargs='+', help="MLP特征提取器的隐藏层维度列表")
parser.add_argument("--output_dim", default=32, type=int, help="特征提取器输出维度，应与回归器输入匹配")

# 学习率相关参数
parser.add_argument("--lr_decay", default=True, type=bool, help="是否使用学习率衰减")
parser.add_argument("--lr_decay_type", default="exponential", type=str, help="学习率衰减类型: exponential, step, cosine")
parser.add_argument("--lr_decay_rate", default=0.98, type=float, help="指数衰减率（越接近1衰减越缓慢）")
parser.add_argument("--lr_step_size", default=5, type=int, help="步长衰减的步长")
parser.add_argument("--lr_min_factor", default=0.4, type=float, help="最小学习率因子（相对于初始学习率）")
parser.add_argument("--lr_decay_start", default=10, type=int, help="开始学习率衰减的轮数（前面轮数保持初始学习率）")
parser.add_argument("--use_best_model", default=True, type=bool, help="是否保存并使用最佳模型")


# 检查点相关参数
parser.add_argument("--resume", action="store_true", help="是否从检查点继续训练")
parser.add_argument("--checkpoint", type=str, default="", help="检查点文件路径")

args = parser.parse_args()

# 设置初始学习率
# 计算学习率
if args.use_fixed_lr:
    # 使用用户指定的固定学习率
    args.eta_init = args.fixed_lr
else:
    # 使用自动计算的学习率，并确保不超过最大值
    auto_lr = 1/np.sqrt(args.num_samples)
    args.eta_init = min(auto_lr, args.max_lr)

args.eta = args.eta_init  # 当前学习率初始化为初始学习率
print(f"学习率设置为: {args.eta_init:.6f}")


# 加载数据集
print(f"正在加载 {args.dataset} 数据集...")
X, Y = data_loader(args)

# 初始化设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 将NumPy数组转换为PyTorch张量并移至正确的设备
print(f"将数据转换为PyTorch张量并移至 {device} 设备...")
for i in range(len(X)):
    X[i] = torch.tensor(X[i], dtype=torch.float32, device=device)
    Y[i] = torch.tensor(Y[i], dtype=torch.float32, device=device)

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

# 初始化参数跟踪列表
w = torch.ones((K, np.prod(gamma.shape)), dtype=torch.float32)
w_loc = torch.ones((K, np.prod(gamma.shape)), dtype=torch.float32)
a = torch.ones((K, 1), dtype=torch.float32)
b = torch.ones((K, 1), dtype=torch.float32)

# 初始化误差跟踪器
mse = torch.zeros((1, 1), dtype=torch.float32)

# 跟踪训练过程中的MSE和MAE
mse_history = []
mae_history = []
rounds_history = []

# 跟踪最小值
best_mse = float('inf')
best_mae = float('inf')
best_model_checkpoint = None
best_model_round = 0

# 将所有张量移到相应设备
w = w.to(device)
w_loc = w_loc.to(device)
a = a.to(device)
b = b.to(device)

# 创建保存模型的目录
# 使用方法名称_客户端数量_全局联邦训练轮数作为文件夹名称
# 创建检查点目录（加上时间戳）
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_type = "FedGPAI" if args.regressor_type == 'linear' else f"FedGPAI_{args.regressor_type}"
checkpoint_dir = f"checkpoints/{model_type}_{args.dataset}_{args.num_clients}_{args.global_rounds}_{timestamp}"
os.makedirs(checkpoint_dir, exist_ok=True)

# 保存模型配置信息
config_file_path = os.path.join(checkpoint_dir, "config.txt")
with open(config_file_path, 'w') as config_file:
    config_file.write(f"回归器类型: {args.regressor_type}\n")
    if args.regressor_type == 'mlp':
        config_file.write(f"MLP隐藏层维度: {args.hidden_dims}\n")
    config_file.write(f"学习率: {args.eta_init}\n")
    config_file.write(f"正则化参数: {args.regularizer}\n")
    config_file.write(f"随机特征数量: {args.num_random_features}\n")
print(f"检查点将保存到: {checkpoint_dir}")

# 创建日志文件
log_file_name = f"FedGPAI_{args.dataset}_{args.num_clients}_{args.global_rounds}.txt"
log_file_path = os.path.join(checkpoint_dir, log_file_name)

# 记录训练起始信息到日志
with open(log_file_path, 'w') as log_file:
    log_file.write(f"===== 训练开始 =====\n")
    log_file.write(f"方法: {'FedGPAI' if args.regressor_type == 'linear' else f'FedGPAI-{args.regressor_type.upper()}'}\n")
    log_file.write(f"数据集: {args.dataset}\n")
    log_file.write(f"回归器类型: {args.regressor_type}\n")
    if args.regressor_type == 'mlp':
        log_file.write(f"MLP隐藏层: {args.hidden_dims}\n")
    log_file.write(f"客户端数量: {args.num_clients}\n")
    log_file.write(f"全局训练轮数: {args.global_rounds}\n")
    log_file.write(f"时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

print(f"日志将保存到: {log_file_path}")

# 初始化变量
start_epoch = 0

# 从检查点恢复训练
if args.resume and args.checkpoint:
    # 如果提供的是完整路径，直接使用；否则在checkpoint_dir中查找
    checkpoint_path = args.checkpoint
    if not os.path.exists(checkpoint_path) and not os.path.isabs(checkpoint_path):
        # 尝试在当前运行的checkpoint_dir中查找
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_path)
    
    if os.path.exists(checkpoint_path):
        print(f"加载检查点: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        start_epoch = checkpoint['epoch']
        w = checkpoint['w']
        w_loc = checkpoint['w_loc']
        a = checkpoint['a']
        b = checkpoint['b']
        
        # 如果有全局模型状态，则需要在创建模型时加载
        global_model_state = checkpoint.get('global_model', None)
        last_mse = checkpoint.get('mse', 0)
        last_mae = checkpoint.get('mae', 0)
        
        print(f"恢复训练从轮次: {start_epoch}")
        print(f"上一轮次MSE: {last_mse:.6f}, MAE: {last_mae:.6f}")
    else:
        print(f"检查点文件不存在: {checkpoint_path}, 从头开始训练")

# 初始化性能评估指标
mse = torch.zeros((args.num_samples, 1), dtype=torch.float32).to(device)
m = torch.zeros((K, args.num_samples, args.global_rounds), dtype=torch.float32).to(device)

print(f"开始联邦学习训练 ({args.global_rounds} 轮全局训练, {args.local_rounds} 轮本地训练)...")

# 执行联邦学习训练过程 (算法3.1第1行: for t ← 0, ..., T - 1 do)
for cc in range(start_epoch, args.global_rounds):
    # 计算当前轮次的学习率衰减因子
    lr_decay_factor = get_lr_decay_factor(cc, args)
    args.eta = args.eta_init * lr_decay_factor
    
    print(f"\n全局轮次 {cc+1}/{args.global_rounds} (学习率: {args.eta:.6f}, 衰减因子: {lr_decay_factor:.4f})")
    
    # 打印初始内存状态
    print_memory_usage("训练前")
    start_time = time.time()
    
    # 清理显存并打印内存状态
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    # 跟踪张量数量和大小
    num_tensors, tensors_size = track_tensors()
    print_memory_usage(f"清理后 轮次{cc+1}")
    print(f"当前活跃张量数: {num_tensors}, 总大小: {tensors_size:.2f} MB")
    
    # 记录本轮耗时
    epoch_time = time.time() - start_time
    print(f"本轮训练耗时: {epoch_time:.2f}秒")
    
    # 使用原始特征维度作为MLP特征提取器的输入维度
    input_dim = X[0].shape[1]
    
    # 使用FedGPAI算法初始化客户端模型（本地模型和混合模型）
    # 参考算法3.1和3.2，第1-5行

    # 我们默认使用MLP特征提取器和MLP回归器(除非特别指定使用线性回归器)
    if args.regressor_type == 'linear':
        print(f"使用线性回归器创建FedGPAI模型...")
        # 创建随机特征矩阵作为向后兼容
        ran_feature = torch.randn(X[0].shape[1], args.num_random_features).to(device)
        local_models, federated_model, hybrid_models = get_FedGPAI(ran_feature, args)
    else:
        print(f"使用MLP特征提取器(隐藏层: {args.extractor_hidden_dims}, 输出维度: {args.output_dim}) ")
        print(f"和MLP回归器(隐藏层: {args.hidden_dims}) 创建FedGPAI模型...")
        local_models, federated_model, hybrid_models = get_FedGPAI_advanced(input_dim, args)
    
    # 如果是从检查点恢复训练的第一个训练轮次，加载模型状态
    if cc == start_epoch and args.resume and args.checkpoint and 'global_model_state' in locals():
        if global_model_state is not None and hasattr(federated_model, 'load_state_dict'):
            try:
                federated_model.load_state_dict(global_model_state)
                print(f"  成功加载全局回归器状态")
            except Exception as e:
                print(f"  加载模型状态失败: {str(e)}")
        else:
            print("  没有可用的模型状态或模型不支持加载状态")
    
    # 创建误差记录器
    e = torch.zeros((args.num_samples, K), dtype=torch.float32).to(device)
    
    # 训练过程 (算法3.1第3行: for 客户端 i ← 1, ‥, N do)
    for i in range(args.num_samples):
        if i % 50 == 0:
            print(f"  评估进度: 样本 {i}/{args.num_samples}")
            
        agg_grad = []  # 聚合梯度（只包含回归器梯度）
        
        # 客户端本地训练
        for j in range(K):
            # 显示客户端本地训练信息
            if i % 50 == 0 and j == 0:
                print(f"    客户端{j+1}开始本地训练: 全局轮次 {cc+1}, 样本 {i}")
                print(f"    将执行 {args.local_rounds} 轮本地训练")
            # 算法3.1第4-5行: 设置全局模型和混合模型
            # θ^t ← (ω_i^t, φ^t) - 全局模型与本地特征提取器的混合 
            # θ_i^t ← (ω_i^t, φ_i^{t-1}) - 本地模型（本地特征提取器和本地回归器）
            
            # 创建全局模型与本地特征提取器的混合模型
            global_hybrid_model = local_models[j].create_hybrid_model(local_models[j].feature_extractor, federated_model.regressor)
            
            # 创建完全本地模型（使用本地特征提取器和上一轮的本地回归器）
            local_hybrid_model = hybrid_models[j]
            
            # 算法3.2第4行: 使用本地特征提取器微调本地回归器
            # 仅允许回归器参数更新，冻结特征提取器
            for local_round in range(args.local_rounds):
                # 冻结特征提取器参数
                for param in local_hybrid_model.feature_extractor.parameters():
                    param.requires_grad = False
                
                # 仅对回归器参数进行微调
                for param in local_hybrid_model.regressor.parameters():
                    param.requires_grad = True
                    
                # 仅训练回归器参数
                local_hybrid_model.fit(X[j][i:i+1, :], Y[j][i:i+1], num_epochs=1, learning_rate=args.eta)
                
                # 恢复特征提取器的参数更新状态
                for param in local_hybrid_model.feature_extractor.parameters():
                    param.requires_grad = True
            
            # 算法3.2第6-12行: 计算两个模型的梯度幅度
            # evaluate_gradient_magnitude实现了算法3.2中的梯度计算
            g_g, g_i = local_models[j].evaluate_gradient_magnitude(
                global_hybrid_model,  # 全局回归器和本地特征提取器的混合模型
                local_hybrid_model,   # 本地回归器和本地特征提取器的混合模型
                X[j][i:i+1, :],      # 本地数据
                Y[j][i:i+1]          # 本地标签
            )
            
            # 算法3.3: 通过model_interpolation实现逐参数自适应插值
            personalized_regressor = local_models[j].model_interpolation(
                global_hybrid_model.regressor,   # 参数1：全局回归器
                local_hybrid_model.regressor,    # 参数2：本地回归器
                g_g,                             # 参数3：全局模型梯度幅度
                g_i                              # 参数4：本地模型梯度幅度
            )
            
            # 算法3.1第10行: 客户端获得初始个性化模型
            local_models[j].regressor = personalized_regressor
            
            # 算法3.1第11行: 本地训练(包括特征提取器和回归器)
            # 使用本地数据的当前样本微调模型
            local_models[j].fit(X[j][i:i+1, :], Y[j][i:i+1], num_epochs=1, learning_rate=args.eta)
            
            # 算法3.3第3-4行: 计算预测误差
            y_pred = hybrid_models[j].predict(X[j][i:i+1, :])
            sq_error = (Y[j][i:i+1] - y_pred) ** 2
            e[i, j] = torch.mean(sq_error)
            
            # 收集梯度用于后续服务器聚合
            # 正确收集个性化后的本地回归器参数(算法3.1第12行)
            if args.regressor_type == 'linear':
                agg_grad.append(local_models[j].regressor.weight)
            else:  # MLP
                agg_grad.append({name: param.data.clone() for name, param in local_models[j].regressor.named_parameters()})
            
            # 这里389-392行的代码似乎与378-380行重复，因为已经计算过误差e[i,j]
            # 如果需要计算累积误差，使用前面计算的y_pred和Y[j][i:i+1]
            # if i == 0:
            #     e[i, j] = torch.mean((y_pred - Y[j][i:i+1])**2)
            # else:
            #     e[i, j] = (1/(i+1)) * ((i*e[i-1, j]) + torch.mean((y_pred - Y[j][i:i+1])**2))
        
        # 改进版算法3.1第9行: 服务器聚合客户端回归器φ^{t+1} ← \frac{1}{N}\sum_{i=1}^N φ̂_i^t
        # 算法3.1第8行: 服务器聚合所有客户端的本地回归器参数
        if args.regressor_type == 'linear':
            new_regressor = torch.mean(torch.stack(agg_grad), dim=0)
            federated_model.regressor = new_regressor
        else:  # MLP回归器 - 参数字典的平均
            # 获取所有参数名称
            param_names = set()
            for grad_dict in agg_grad:
                param_names.update(grad_dict.keys())
                
            # 对每个参数计算平均值
            avg_params = {}
            for name in param_names:
                param_stack = []
                for grad_dict in agg_grad:
                    if name in grad_dict:
                        param_stack.append(grad_dict[name])
                if param_stack:
                    avg_params[name] = torch.mean(torch.stack(param_stack), dim=0)
            
            # 更新全局模型参数
            for name, param in federated_model.regressor.named_parameters():
                if name in avg_params:
                    param.data.copy_(avg_params[name])
    
    # 计算平均误差
    mse = (1/(cc+1)) * ((cc*mse)+torch.reshape(torch.mean(e, dim=1), (-1, 1)))
    
    # 每轮记录MSE和MAE
    current_mse = mse[-1].item()
    current_mae = torch.mean(torch.sqrt(mse[-1])).item()
    mse_history.append(current_mse)
    mae_history.append(current_mae)
    rounds_history.append(cc+1)
    
    # 更新最小值
    if current_mse < best_mse:
        best_mse = current_mse
        best_mae = current_mae
        best_model_round = cc + 1
        
        # 如果启用了最佳模型保存功能，则保存当前模型为最佳模型
        if args.use_best_model:
            best_model_checkpoint = {
                'epoch': cc + 1,
                'global_model': federated_model.state_dict() if hasattr(federated_model, 'state_dict') else None,
                'regressor_type': args.regressor_type,
                'hidden_dims': args.hidden_dims if args.regressor_type == 'mlp' else None,
                # 添加MLP特征提取器相关参数
                'extractor_hidden_dims': args.extractor_hidden_dims,
                'output_dim': args.output_dim,
                'w': w.clone() if isinstance(w, torch.Tensor) else w.copy(),
                'w_loc': w_loc.clone() if isinstance(w_loc, torch.Tensor) else w_loc.copy(),
                'a': a.clone() if isinstance(a, torch.Tensor) else a.copy(),
                'b': b.clone() if isinstance(b, torch.Tensor) else b.copy(),
                'mse': best_mse,
                'mae': best_mae
            }
            
            # 保存最佳模型
            best_model_path = os.path.join(checkpoint_dir, "best_model.pt")
            torch.save(best_model_checkpoint, best_model_path)
            print(f"  发现新的最佳模型! MSE: {best_mse:.6f}, MAE: {best_mae:.6f}, 已保存到: {best_model_path}")
    
    # 每轮计算并输出一次MAE和MSE
    if (cc+1) % 1 == 0 or cc == 0:
        print(f"\n  Round {cc+1} - MSE: {current_mse:.6f}, MAE: {current_mae:.6f}")
        
        # 将结果写入日志文件
        with open(log_file_path, 'a') as log_file:
            log_file.write(f"轮次 {cc+1}/{args.global_rounds}\n")
            log_file.write(f"MSE: {current_mse:.6f}\n")
            log_file.write(f"MAE: {current_mae:.6f}\n")
            log_file.write(f"时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # 保存模型
        checkpoint_filename = f"epoch_{cc+1}.pt"
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_filename)
        checkpoint = {
            'epoch': cc + 1,
            'global_model': federated_model.state_dict() if hasattr(federated_model, 'state_dict') else None,
            'w': w,
            'w_loc': w_loc,
            'a': a,
            'b': b,
            'mse': mse[-1].item(),
            'mae': current_mae
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"  模型已保存到: {checkpoint_path}")
        print(f"  结果已记录到: {log_file_path}")
    
    # 每轮结束后清理内存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    # 释放本轮不再需要的大型变量
    del e
    if 'ran_feature' in locals():
        del ran_feature
    if 'agg_grad' in locals():
        del agg_grad
    
    # 清理显存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# 绘制和保存MSE曲线
plt.figure(figsize=(10, 6))
plt.plot(rounds_history, mse_history, 'b-o')
plt.xlabel('Round')
plt.ylabel('Mean Squared Error (MSE)')
plt.title('FedGPAI Training MSE over Rounds')
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
plt.title('FedGPAI Training MAE over Rounds')
plt.grid(True)
plt.tight_layout()
mae_plot_path = os.path.join(checkpoint_dir, 'mae_curve.png')
plt.savefig(mae_plot_path)
plt.close()

# 打印最终结果
final_mse = mse[-1].item()
final_mae = torch.mean(torch.sqrt(mse[-1])).item()

# 训练结束，如果启用了使用最佳模型功能，则加载最佳模型
if args.use_best_model and best_model_checkpoint is not None:
    print(f"\n加载最佳模型（轮次 {best_model_round}）...")
    # 从最佳模型恢复状态
    w = best_model_checkpoint['w']
    w_loc = best_model_checkpoint['w_loc']
    a = best_model_checkpoint['a']
    b = best_model_checkpoint['b']
    if hasattr(federated_model, 'load_state_dict') and best_model_checkpoint['global_model'] is not None:
        federated_model.load_state_dict(best_model_checkpoint['global_model'])
    print(f"已加载最佳模型 (MSE: {best_mse:.6f}, MAE: {best_mae:.6f})")

# 打印最终结果
final_mse = mse_history[-1]
final_mae = mae_history[-1]
print(f"\n训练完成! 总轮数: {args.global_rounds}")
print(f"最终 MSE: {final_mse:.6f}")
print(f"最终 MAE: {final_mae:.6f}")
print(f"最佳 MSE: {best_mse:.6f} (轮次 {best_model_round})")
print(f"最佳 MAE: {best_mae:.6f} (轮次 {best_model_round})")
print(f'FedGPAI standard deviation: {torch.std(torch.mean(torch.mean(m, dim=2), dim=1)).item():.6f}')
print(f'MSE curve saved to: {mse_plot_path}')
print(f'MAE curve saved to: {mae_plot_path}')

# 记录最终结果到日志
with open(log_file_path, 'a') as log_file:
    log_file.write(f"===== Training Completed =====\n")
    log_file.write(f"Final MSE: {final_mse:.6f}\n")
    log_file.write(f"Final MAE: {final_mae:.6f}\n")
    log_file.write(f"Best MSE: {best_mse:.6f}\n")
    log_file.write(f"Best MAE: {best_mae:.6f}\n")
    log_file.write(f"Standard Deviation: {torch.std(torch.mean(torch.mean(m, dim=2), dim=1)).item():.6f}\n")
    log_file.write(f"MSE curve saved to: {mse_plot_path}\n")
    log_file.write(f"MAE curve saved to: {mae_plot_path}\n")
    log_file.write(f"Completion Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
