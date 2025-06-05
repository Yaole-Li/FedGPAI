import numpy as np
import argparse
import torch
import os
import gc
from lib.datasets.data_loader import data_loader
from lib.FedGPAI.FedGPAI_regression import FedGPAI_regression

# 命令行参数设置
parser = argparse.ArgumentParser()

# 数据集和任务相关参数
parser.add_argument("--dataset", default='Air', type=str, help="数据集名称")
parser.add_argument("--task", default='regression', type=str, help="任务类型")

# 客户端相关参数
parser.add_argument("--num_clients", default=40, type=int, help="客户端数量")
parser.add_argument("--num_samples", default=250, type=int, help="每个客户端的样本数量")
parser.add_argument("--test_ratio", default=0.2, type=float, help="测试集比例")

# 模型相关参数
parser.add_argument("--num_random_features", default=100, type=int, help="随机特征数量")
parser.add_argument("--regularizer", default=1e-6, type=float, help="正则化参数")
parser.add_argument("--global_rounds", default=20, type=int, help="全局联邦训练轮数")
parser.add_argument("--local_rounds", default=10, type=int, help="本地训练轮数")
parser.add_argument("--train_head_epochs", default=5, type=int, help="训练回归器的轮数")

# 检查点相关参数
parser.add_argument("--resume", action="store_true", help="是否从检查点继续训练")
parser.add_argument("--checkpoint", type=str, default="", help="检查点文件路径")
parser.add_argument("--save_dir", type=str, default="checkpoints/fedrep", help="保存模型的目录")

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

# 设置随机核参数
print("初始化随机特征...")
gamma = []
num_rbf = 3
for i in range(num_rbf):
    gamma.append(10**(i-1))
gamma = np.array(gamma)

# 设置随机特征数量
n_components = args.num_random_features

# 创建保存模型的目录
os.makedirs(args.save_dir, exist_ok=True)

# 初始化变量
start_epoch = 0

# 初始化性能评估指标
mse = torch.zeros((args.num_samples, 1), dtype=torch.float32).to(device)
m = torch.zeros((K, args.num_samples, args.global_rounds), dtype=torch.float32).to(device)


def get_fedrep_models(random_features, args):
    """
    获取FedRep模型实例
    
    Args:
        random_features: 随机特征
        args: 参数对象
        
    Returns:
        local_models: 本地模型列表
        global_model: 全局模型
    """
    # 创建全局模型（只包含回归器部分）
    global_model = FedGPAI_regression(args.regularizer, random_features, args.eta, args.num_clients, is_global=True)
    
    # 创建每个客户端的本地模型
    local_models = []
    
    for i in range(args.num_clients):
        # 创建本地模型，包含本地特征提取器和本地回归器
        local_model = FedGPAI_regression(args.regularizer, random_features, args.eta, args.num_clients, is_global=False)
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
    
    # 生成随机特征
    ran_feature = torch.zeros((N, n_components, gamma.shape[0]), dtype=torch.float32)
    for i in range(num_rbf):
        ran_feature[:, :, i] = torch.randn(N, n_components) * torch.sqrt(torch.tensor(1/gamma[i], dtype=torch.float32))
        
    # 移动到相应设备
    ran_feature = ran_feature.to(device)
    
    # 获取FedRep模型
    local_models, global_model = get_fedrep_models(ran_feature, args)
    
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
            local_model.regressor = global_model.regressor.clone()
            
            # 将数据转换为PyTorch张量
            x_j = torch.tensor(X[j][i:i+1, :], dtype=torch.float32).to(device)
            y_j = torch.tensor(Y[j][i], dtype=torch.float32).to(device)
            
            # FedRep本地训练（分两个阶段）
            for local_round in range(args.local_rounds):
                # 模型预测
                outputs, _, X_features = local_model.predict(x_j, None)
                
                # 计算损失
                loss = (outputs - y_j)**2
                
                # 手动计算梯度
                if local_round < args.train_head_epochs:
                    # 阶段1：只训练回归器，冻结特征提取器
                    X_features_t = X_features.t()
                    regressor_grad = (2.0 / x_j.shape[0]) * torch.matmul(X_features_t, (outputs - y_j))
                    
                    # 更新回归器
                    local_model.regressor -= args.eta * regressor_grad
                else:
                    # 阶段2：只训练特征提取器，冻结回归器
                    input_features_grad = (2.0 / x_j.shape[0]) * torch.matmul((outputs - y_j), local_model.regressor.t())
                    feature_extractor_grad = torch.matmul(x_j.t(), input_features_grad)
                    
                    # 更新特征提取器
                    local_model.feature_extractor -= args.eta * feature_extractor_grad
            
            # 训练结束后，使用本地模型进行预测
            outputs, _, _ = local_model.predict(x_j, None)
            
            # 记录当前轮次的均方误差
            current_mse = (outputs - y_j)**2
            m[j, i, cc] = current_mse
            
            # 计算累积误差
            if i == 0:
                e[i, j] = current_mse
            else:
                e[i, j] = (1/(i+1)) * ((i*e[i-1, j]) + current_mse)
            
            # 收集回归器参数用于全局聚合
            all_regressors.append(local_model.regressor.clone())
        
        # 全局模型聚合（只聚合回归器）
        if all_regressors:
            # 计算所有回归器的平均值
            avg_regressor = torch.zeros_like(global_model.regressor)
            for regressor in all_regressors:
                avg_regressor += regressor
            avg_regressor /= len(all_regressors)
            
            # 更新全局回归器
            global_model.regressor = avg_regressor.clone()
    
    # 计算平均误差
    mse = (1/(cc+1)) * ((cc*mse)+torch.reshape(torch.mean(e, dim=1), (-1, 1)))
    
    # 每5轮计算并输出一次MAE
    if (cc+1) % 5 == 0 or cc == 0:
        current_mae = torch.mean(torch.sqrt(mse[-1])).item()
        print(f"\n  当前轮次 {cc+1} 的MAE为: {current_mae:.6f}")
        
        # 保存模型
        checkpoint = {
            'epoch': cc + 1,
            'global_model': global_model.state_dict() if hasattr(global_model, 'state_dict') else None,
            'mse': mse[-1].item(),
            'mae': current_mae
        }
        torch.save(checkpoint, f"{args.save_dir}/fedrep_checkpoint_epoch_{cc+1}.pt")
        print(f"  模型已保存到: {args.save_dir}/fedrep_checkpoint_epoch_{cc+1}.pt")
    
    # 清理显存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

# 打印最终结果
print('FedRep的MSE为：%s' % mse[-1].item())
print('FedRep的标准差为：%s' % torch.std(torch.mean(torch.mean(m, dim=2), dim=1)).item())
