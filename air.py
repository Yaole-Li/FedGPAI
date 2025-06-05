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
parser.add_argument("--num_clients", default=40, type=int, help="客户端数量")
parser.add_argument("--num_samples", default=250, type=int, help="每个客户端的样本数量")
parser.add_argument("--test_ratio", default=0.2, type=float, help="测试集比例")

# 模型相关参数
parser.add_argument("--num_random_features", default=100, type=int, help="随机特征数量")
parser.add_argument("--regularizer", default=1e-6, type=float, help="正则化参数")
parser.add_argument("--global_rounds", default=20, type=int, help="全局联邦训练轮数")
parser.add_argument("--local_rounds", default=5, type=int, help="本地训练轮数")

# 检查点相关参数
parser.add_argument("--resume", action="store_true", help="是否从检查点继续训练")
parser.add_argument("--checkpoint", type=str, default="", help="检查点文件路径")

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

# 将所有张量移到相应设备
w = w.to(device)
w_loc = w_loc.to(device)
a = a.to(device)
b = b.to(device)

# 创建保存模型的目录
os.makedirs("checkpoints", exist_ok=True)

# 初始化变量
start_epoch = 0

# 从检查点恢复训练
if args.resume and args.checkpoint:
    if os.path.exists(args.checkpoint):
        print(f"加载检查点: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint)
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
        print(f"检查点文件不存在: {args.checkpoint}, 从头开始训练")

# 初始化性能评估指标
mse = torch.zeros((args.num_samples, 1), dtype=torch.float32).to(device)
m = torch.zeros((K, args.num_samples, args.global_rounds), dtype=torch.float32).to(device)

print(f"开始联邦学习训练 ({args.global_rounds} 轮全局训练, {args.local_rounds} 轮本地训练)...")

# 执行联邦学习训练过程 (算法3.1第1行: for t ← 0, ..., T - 1 do)
for cc in range(start_epoch, args.global_rounds):
    print(f"\n全局轮次 {cc+1}/{args.global_rounds}")
    
    # 生成随机特征
    ran_feature = torch.zeros((N, n_components, gamma.shape[0]), dtype=torch.float32)
    for i in range(num_rbf):
        ran_feature[:, :, i] = torch.randn(N, n_components) * torch.sqrt(torch.tensor(1/gamma[i], dtype=torch.float32))
        
    # 移动到相应设备
    ran_feature = ran_feature.to(device)
    
    # 获取FedGPAI模型 (改进版算法3.1第2行: 服务器发送全局回归器φ^t给所有参与的客户端)
    alg_loc, alg, alg_hybrid = get_FedGPAI(ran_feature, args)
    
    # 如果是从检查点恢复训练的第一个训练轮次，加载模型状态
    if cc == start_epoch and args.resume and args.checkpoint and 'global_model_state' in locals():
        if global_model_state is not None and hasattr(alg, 'load_state_dict'):
            try:
                alg.load_state_dict(global_model_state)
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
            # θ^t ← (ω^t, φ^t) - 全局模型
            # θ_i^t ← (ω^t, φ_i^{t-1}) - 混合模型
            
            # 更新混合模型，用本地特征提取器和全局回归器
            hybrid_model = alg_loc[j].create_hybrid_model(alg_loc[j].feature_extractor, alg.regressor)
            alg_hybrid[j] = hybrid_model
            
            # 算法3.2: 基于梯度幅度的参数重要性评估
            # 首先对混合模型的回归器进行微调（算法3.2第4行）
            
            # 当前客户端本地训练开始
            local_train_start = True
            
            # 本地训练多轮
            for local_round in range(args.local_rounds):
                # 混合模型微调
                f_RF_hybrid, f_RF_p_hybrid, X_features_hybrid = alg_hybrid[j].predict(X[j][i:i+1, :], None)
                loss_hybrid = (f_RF_hybrid - Y[j][i])**2
                # 通过梯度下降微调混合模型的回归器
                _, hybrid_grad = alg_hybrid[j].local_update(f_RF_p_hybrid, Y[j][i], torch.ones_like(w[j:j+1, :]), X_features_hybrid)
            
            # 算法3.2第6-12行: 计算全局模型和混合模型梯度幅度
            g_g, g_i = alg.evaluate_gradient_magnitude(alg, alg_hybrid[j], X[j][i:i+1, :], Y[j][i])
            
            # 算法3.3: 基于逐参数自适应插值的个性化回归器优化
            personalized_regressor = alg.model_interpolation(g_g, g_i, alg.regressor, alg_hybrid[j].regressor)
            
            # 算法3.1第10行: 客户端 i 获得初始模型θ̂_i^t = (ω^t, φ̂_i^t)
            alg_loc[j].regressor = personalized_regressor
            
            # 算法3.1第11行: θ̂_i^t ← θ̂_i^t - α∇_{θ_i}L(θ̂_i^t; D_i) - 本地训练
            # 将NumPy数组转换为PyTorch张量并移动到指定设备
            x_j = torch.tensor(X[j][i:i+1, :], dtype=torch.float32).to(device)
            y_j = torch.tensor(Y[j][i], dtype=torch.float32).to(device)
            
            # 使用全局模型预测和更新
            f_RF_fed, f_RF_p, X_features = alg.predict(x_j, w[j:j+1, :])
            w[j:j+1, :], local_grad = alg.local_update(f_RF_p, y_j, w[j:j+1, :], X_features)
            
            # 使用本地模型预测和更新 (这里是个性化模型的训练)
            f_RF_loc, f_RF_p, X_features = alg_loc[j].predict(x_j, w_loc[j:j+1, :])
            w_loc[j:j+1, :], local_grad_loc = alg_loc[j].local_update(f_RF_p, y_j, w_loc[j:j+1, :], X_features)
            
            # 组合预测
            f_RF = (a[j, 0]*f_RF_fed + b[j, 0]*f_RF_loc)/(a[j, 0]+b[j, 0])
            
            # 计算损失并更新权重
            l_fed = (f_RF_fed-Y[j][i])**2
            l_loc = (f_RF_loc-Y[j][i])**2
            
            # 更安全的广播处理
            # 先计算指数项
            exp_term_fed = torch.exp(-args.eta * l_fed)
            exp_term_loc = torch.exp(-args.eta * l_loc)
            
            # 确保指数项有正确的形状以便广播
            if not isinstance(exp_term_fed, torch.Tensor):
                exp_term_fed = torch.tensor(exp_term_fed, device=a.device)
            if exp_term_fed.dim() == 0:  # 标量形状
                exp_term_fed = exp_term_fed.reshape(1, 1)
                
            if not isinstance(exp_term_loc, torch.Tensor):
                exp_term_loc = torch.tensor(exp_term_loc, device=b.device)
            if exp_term_loc.dim() == 0:  # 标量形状
                exp_term_loc = exp_term_loc.reshape(1, 1)
            
            # 执行广播兼容的乘法操作 
            a[j, 0] = a[j, 0] * exp_term_fed
            b[j, 0] = b[j, 0] * exp_term_loc
            
            # 本地模型更新
            alg_loc[j].global_update([local_grad_loc])
            
            # 改进版算法3.1第8行: 客户端 i 只发送回归器参数φ̂_i^t至中央服务器
            # 收集回归器梯度用于全局模型更新
            agg_grad.append(local_grad)
            
            # 记录当前轮次的均方误差
            m[j, i, cc] = (f_RF-y_j)**2
            
            # 计算累积误差
            if i == 0:
                e[i, j] = (f_RF-y_j)**2
            else:
                e[i, j] = (1/(i+1)) * ((i*e[i-1, j])+((f_RF-y_j)**2))
        
        # 改进版算法3.1第9行: 服务器聚合客户端回归器φ^{t+1} ← \frac{1}{N}\sum_{i=1}^N φ̂_i^t
        # 只聚合回归器参数
        alg.global_update(agg_grad)
    
    # 计算平均误差
    mse = (1/(cc+1)) * ((cc*mse)+torch.reshape(torch.mean(e, dim=1), (-1, 1)))
    
    # 每5轮计算并输出一次MAE
    if (cc+1) % 5 == 0 or cc == 0:
        current_mae = torch.mean(torch.sqrt(mse[-1])).item()
        print(f"\n  当前轮次 {cc+1} 的MAE为: {current_mae:.6f}")
        
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
        torch.save(checkpoint, f"checkpoints/fedgpai_checkpoint_epoch_{cc+1}.pt")
        print(f"  模型已保存到: checkpoints/fedgpai_checkpoint_epoch_{cc+1}.pt")
    
    # 清理显存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

# 打印最终结果
print('FedGPAI的MSE为：%s' % mse[-1].item())
print('FedGPAI的标准差为：%s' % torch.std(torch.mean(torch.mean(m, dim=2), dim=1)).item())
