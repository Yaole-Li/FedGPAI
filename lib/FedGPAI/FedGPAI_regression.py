import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, List

# 检查是否有可用的GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class FedGPAI_regression:
    """
    FedGPAI回归模型类，实现算法3.1-3.3，使用PyTorch实现特征提取器和回归器分离的联邦学习
    """
    def __init__(self, lam, rf_feature, eta, num_clients, is_global=False):
        """
        初始化 FedGPAI 回归模型
        
        Args:
            lam: 正则化参数
            rf_feature: 随机特征
            eta: 学习率
            num_clients: 客户端数量
            is_global: 是否为全局模型，用于区分全局模型和本地模型
        """
        if isinstance(rf_feature, np.ndarray):
            # 如果传入的是numpy数组，转换成torch.Tensor并移动到GPU
            self.rf_feature = torch.from_numpy(rf_feature).float().to(device)
        else:
            self.rf_feature = rf_feature.to(device)
            
        # 初始化特征提取器和回归器，并移动到GPU
        self.feature_extractor = nn.Parameter(torch.from_numpy(rf_feature).float().to(device))
        self.regressor = nn.Parameter(torch.zeros(rf_feature.shape[1], 1).to(device))
        
        # 初始化参数
        self.lam = lam  # 正则化参数
        self.eta = eta  # 学习率
        self.num_clients = num_clients  # 客户端数量
        
        # 设置核数量（用于判断回归器维度）
        self.num_kernels = 1  # 默认为1，根据实际情况可能会根据回归器形状调整
        
        # 全局模型和本地模型区分
        self.is_global = is_global  # 算法3.1第4-5行，区分全局模型和本地模型
        
        # 初始化全局和本地权重
        self.global_weights = torch.ones((1, self.num_kernels), dtype=torch.float32).to(device) / self.num_kernels
        
        # 混合模型状态记录，表示当前模型是否是混合模型(全局特征提取器+本地回归器)
        
    def create_hybrid_model(self, global_feature_extractor, local_regressor):
        """
        创建混合模型(全局特征提取器+本地回归器) (算法3.1第5行)
        
        Args:
            global_feature_extractor: 全局模型的特征提取器
            local_regressor: 本地模型的回归器
            
        Returns:
            hybrid_model: 混合模型
        """
        hybrid_model = FedGPAI_regression(self.lam, self.rf_feature.numpy(), self.eta, self.num_clients, is_global=False)
        
        # 使用全局特征提取器
        hybrid_model.feature_extractor = global_feature_extractor.clone().detach()
        
        # 使用本地回归器
        hybrid_model.regressor = local_regressor.clone().detach()
        
        return hybrid_model
        
    def extract_features(self, X):
        """
        使用特征提取器从输入数据提取特征
        
        Args:
            X: 输入数据
            
        Returns:
            X_features: 提取的特征
        """
        # 确保X是Torch张量并移入GPU
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32).to(device)
        elif X.device != device:
            X = X.to(device)
            
        # 简化特征提取过程
        X_features = torch.matmul(X, self.feature_extractor)
        return X_features
    
    def predict(self, X, w=None):
        """
        使用特征提取器和回归器进行预测
        
        Args:
            X: 输入数据
            w: 模型权重 (可选)
        
        Returns:
            f_RF_fed: 联邦全局模型预测值
            f_RF_p: 预测值
            X_features: 特征
        """
        # 确保X是Torch张量并移入GPU
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32).to(device)
        elif X.device != device:
            X = X.to(device)
            
        # 使用特征提取器从输入数据提取特征
        X_features = self.extract_features(X)
        
        # 使用回归器进行预测
        if w is not None:
            # 如果给定权重，则使用提供的权重
            f_RF_p = torch.zeros(X.shape[0], 1).to(device)
            for j in range(X.shape[0]):
                # 确保维度一致性的矩阵乘法
                f_RF_p[j, 0] = torch.matmul(X_features[j, :], self.regressor)
            return f_RF_p, f_RF_p, X_features
        else:
            # 否则使用默认模型参数
            f_RF_p = torch.matmul(X_features, self.regressor)
            return f_RF_p, f_RF_p, X_features
    
    def evaluate_gradient_magnitude(self, global_model, local_model, X, Y):
        """
        基于梯度幅度的参数重要性评估 (算法3.2)
        
        Args:
            global_model: 全局模型
            local_model: 混合模型 (全局特征提取器 + 本地回归器)
            X: 输入数据
            Y: 真实标签
            
        Returns:
            g_g: 全局模型回归器梯度幅度
            g_i: 混合模型回归器梯度幅度
        """
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)
        if not isinstance(Y, torch.Tensor):
            Y = torch.tensor(Y, dtype=torch.float32)
            
        # 算法3.2第4行: 回归器微调
        # 对混合模型的回归器进行微调
        # (这一步在air.py和wec.py的训练循环中实现)
        
        # 算法3.2第6-8行: 全局模型处理
        # 使用全局模型计算损失
        f_RF_global, f_RF_p_global, X_features_global = global_model.predict(X)
        loss_global = (f_RF_global - Y)**2  # 均方误差损失
        
        # 计算全局回归器梯度
        global_grad = torch.zeros_like(global_model.regressor)
        for j in range(self.num_kernels):
            # 计算梯度 (∂loss/∂φ)
            global_grad[:, j] = 2 * (f_RF_p_global[j, 0] - Y) * X_features_global[j, :].t()
            # 添加L2正则化项梯度
            global_grad[:, j] += 2 * self.lam * global_model.regressor[:, j]
        
        # 计算全局模型回归器梯度L2范数 (梯度幅度)
        g_g = torch.norm(global_grad, p=2)
        
        # 算法3.2第9-12行: 混合模型处理
        # 使用混合模型计算损失
        f_RF_local, f_RF_p_local, X_features_local = local_model.predict(X)
        loss_local = (f_RF_local - Y)**2  # 均方误差损失
        
        # 计算混合模型回归器梯度
        local_grad = torch.zeros_like(local_model.regressor)
        for j in range(self.num_kernels):
            # 计算梯度 (∂loss/∂φ)
            local_grad[:, j] = 2 * (f_RF_p_local[j, 0] - Y) * X_features_local[j, :].t()
            # 添加L2正则化项梯度
            local_grad[:, j] += 2 * self.lam * local_model.regressor[:, j]
            
        # 计算混合模型回归器梯度L2范数 (梯度幅度)
        g_i = torch.norm(local_grad, p=2)
        
        return g_g, g_i
    
    def local_update(self, f_RF_p, y, w, X_features):
        """
        本地模型更新
        
        Args:
            f_RF_p: 模型预测值
            y: 真实标签
            w: 模型权重
            X_features: 特征
            
        Returns:
            w: 更新后的模型权重
            local_grad: 本地梯度
        """
        # 确保数据在GPU上
        if isinstance(y, (int, float)):
            y = torch.tensor([y], dtype=torch.float32).to(device)
        elif not isinstance(y, torch.Tensor):
            y = torch.tensor(y, dtype=torch.float32).to(device)
        elif y.device != device:
            y = y.to(device)
            
        if not isinstance(w, torch.Tensor):
            w = torch.tensor(w, dtype=torch.float32).to(device)
        elif w.device != device:
            w = w.to(device)
        
        # 计算损失
        loss = (f_RF_p - y)**2
        
        # 计算梯度
        local_grad = 2 * (f_RF_p - y) * X_features
        
        # 更新回归器参数
        self.regressor.data = self.regressor.data - self.eta * local_grad.mean(0).unsqueeze(1) - self.lam * self.regressor.data
        
        # 更新模型权重
        w_new = w * torch.exp(-self.eta * loss)
        
        # 返回更新后的权重和本地梯度
        return w_new, local_grad
    
    def global_update(self, agg_grad):
        """
        全局模型更新
        
        Args:
            agg_grad: 聚合梯度
        """
        # 如果不是全局模型，则不执行全局更新
        if not self.is_global:
            return
            
        # 将聚合梯度转换为Torch张量并移到GPU
        if isinstance(agg_grad, list):
            processed_grads = []
            for grad in agg_grad:
                if isinstance(grad, torch.Tensor):
                    if grad.device != device:
                        processed_grads.append(grad.to(device))
                    else:
                        processed_grads.append(grad)
                else:
                    processed_grads.append(torch.tensor(grad, dtype=torch.float32).to(device))
            agg_grad = torch.stack(processed_grads)
        elif not isinstance(agg_grad, torch.Tensor):
            agg_grad = torch.tensor(agg_grad, dtype=torch.float32).to(device)
        elif agg_grad.device != device:
            agg_grad = agg_grad.to(device)
            
        # 处理收集的梯度并使用于GPU
        try:
            # 平均梯度
            avg_grad = torch.mean(agg_grad, dim=0)
            
            # 直接更新回归器参数
            self.regressor.data -= self.eta * avg_grad.mean(0).unsqueeze(1) + self.lam * self.regressor.data
        except Exception as e:
            print(f"\n全局更新过程中遇到问题: {e}")
            print(f"agg_grad shape: {agg_grad.shape if isinstance(agg_grad, torch.Tensor) else 'not tensor'}")
            print(f"regressor shape: {self.regressor.shape}")
            # 提供备选更新方式
            if isinstance(agg_grad, torch.Tensor):
                # 尝试另一种更新方式
                flattened_grad = agg_grad.reshape(-1, agg_grad.shape[-1])
                avg_grad = torch.mean(flattened_grad, dim=0).unsqueeze(1)
                self.regressor.data -= self.eta * avg_grad + self.lam * self.regressor.data
    
    def feature_extractor_update(self, agg_fe_grad):
        """
        特征提取器更新
        
        Args:
            agg_fe_grad: 聚合的特征提取器梯度列表
        """
        # 确保梯度是PyTorch张量
        if not all(isinstance(grad, torch.Tensor) for grad in agg_fe_grad):
            agg_fe_grad = [torch.tensor(grad, dtype=torch.float32) if not isinstance(grad, torch.Tensor) else grad 
                           for grad in agg_fe_grad]
            
        fe_update = torch.zeros_like(self.feature_extractor)
        
        # 聚合特征提取器梯度
        for i in range(len(agg_fe_grad)):
            fe_update += (agg_fe_grad[i] / len(agg_fe_grad))
        
        # 更新特征提取器参数
        self.feature_extractor -= self.eta * fe_update
        
    def model_interpolation(self, g_g, g_i, global_regressor, local_regressor):
        """
        基于逐参数自适应插值的个性化回归器优化 (算法3.3)
        
        Args:
            g_g: 全局模型回归器梯度幅度
            g_i: 混合模型回归器梯度幅度
            global_regressor: 全局模型回归器
            local_regressor: 本地模型回归器
            
        Returns:
            personalized_regressor: 个性化回归器
        """
        if not isinstance(g_g, torch.Tensor):
            g_g = torch.tensor(g_g, dtype=torch.float32)
        if not isinstance(g_i, torch.Tensor):
            g_i = torch.tensor(g_i, dtype=torch.float32)
        
        # 算法3.3第3-4行: 计算L2范数
        # 梯度幅度g_g和g_i已经是范数，所以不需要再计算
        
        # 创建用于存储个性化回归器参数的张量
        personalized_regressor = torch.zeros_like(global_regressor)
        
        # 获取回归器参数的形状
        regressor_shape = global_regressor.shape
        
        # 扁平化处理便于遍历每个参数
        g_g_flat = g_g.view(-1)
        g_i_flat = g_i.view(-1)
        global_regressor_flat = global_regressor.view(-1)
        local_regressor_flat = local_regressor.view(-1)
        personalized_regressor_flat = personalized_regressor.view(-1)
        
        # 算法3.3第5-13行: 遍历每个参数
        epsilon = 1e-8  # 防止除零
        
        # 对所有参数统一进行归一化处理
        g_g_norm = g_g / (g_g.norm() + epsilon)
        g_i_norm = g_i / (g_i.norm() + epsilon)
        
        for j in range(len(personalized_regressor_flat)):
            # 算法3.3第9-10行: 计算插值权重
            alpha_j = 1.0 - g_i_norm / (g_i_norm + g_g_norm + epsilon)
            
            # 算法3.3第12行: 调整融合比例
            personalized_regressor_flat[j] = alpha_j * local_regressor_flat[j] + (1 - alpha_j) * global_regressor_flat[j]
        
        # 恢复原始形状
        personalized_regressor = personalized_regressor_flat.view(regressor_shape)
        
        return personalized_regressor
