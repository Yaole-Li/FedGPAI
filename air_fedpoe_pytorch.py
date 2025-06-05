import argparse
import torch
import numpy as np
from lib.datasets.data_loader import data_loader
from lib.FedPOE.get_FedPOE_pytorch import get_FedPOE

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
w = w.to(device)
w_loc = w_loc.to(device)
a = a.to(device)
b = b.to(device)

# 初始化评估指标
mse = torch.zeros((args.num_samples, 1), dtype=torch.float32).to(device)
m = torch.zeros((K, args.num_samples, args.global_rounds), dtype=torch.float32).to(device)

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
            
            # 指数加权权重更新
            a[j, 0] *= torch.exp(-args.eta * l_fed)
            b[j, 0] *= torch.exp(-args.eta * l_loc)
            
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

# 打印最终结果
print(f'\nFedPOE的MSE为: {mse[-1].item()}')
print(f'FedPOE的标准差为: {torch.std(torch.mean(torch.mean(m, dim=2), dim=1)).item()}')
