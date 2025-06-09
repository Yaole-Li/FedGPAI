import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, List
from lib.FedGPAI.models import create_regressor, LinearRegressor, MLPRegressor

# 检查是否有可用的GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class FedGPAI_advanced:
    """
    增强版FedGPAI模型类，支持可选的回归器类型(线性或MLP)
    实现算法3.1-3.3的增强版本，特征提取器和回归器分离的联邦学习
    """
    def __init__(self, lam=1e-5, rf_feature=None, eta=1e-4, regressor_type='linear', 
                 hidden_dims=None, num_clients=1, is_global=False):
        """初始化增强版FedGPAI模型
        
        参数:
            lam (float): 正则化参数
            rf_feature (numpy.ndarray 或 torch.Tensor): 随机特征张量
            eta (float): 学习率
            regressor_type (str): 回归器类型，'linear'或'mlp'
            hidden_dims (List[int]): MLP回归器的隐藏层大小，仅当regressor_type='mlp'时使用
            num_clients (int): 客户端数量
            is_global (bool): 是否为全局模型
        """
        # 设置设备，优先使用GPU
        self.device = device
        self.input_feature_dim = None
        self.regressor_type = regressor_type
        self.hidden_dims = hidden_dims if hidden_dims else [64, 32]
        
        # 如果是numpy数组，转换为torch张量
        if isinstance(rf_feature, np.ndarray):
            self.rf_feature = torch.from_numpy(rf_feature).float().to(self.device)
        else:
            self.rf_feature = rf_feature
            
        # 确定随机特征维度
        if rf_feature is not None:
            if len(rf_feature.shape) == 3:  # 对于3D特征
                n_samples, n_features, n_rf = rf_feature.shape
                self.input_feature_dim = self.input_feature_dim or n_features
                self.n_rf = n_rf
            else:  # 对于2D特征
                n_rf = rf_feature.shape[0] if len(rf_feature.shape) > 0 else 100
                self.n_rf = n_rf
                if len(rf_feature.shape) > 1:
                    self.input_feature_dim = self.input_feature_dim or rf_feature.shape[1]
        else:
            self.n_rf = 100
        
        # 如果仍未确定输入特征维度，设置默认值
        self.input_feature_dim = self.input_feature_dim or self.n_rf
        
        # 初始化特征提取器为适当维度
        self.feature_extractor = torch.randn(self.input_feature_dim, self.n_rf, dtype=torch.float32).to(self.device)
        self.feature_extractor.requires_grad_(True)
        
        # 初始化回归器
        self.regressor = create_regressor(
            regressor_type=regressor_type,
            input_dim=self.n_rf,
            hidden_dims=self.hidden_dims
        )
        
        # 初始化参数
        self.lam = lam  # 正则化参数
        self.eta = eta  # 学习率
        self.num_clients = num_clients  # 客户端数量
        
        # 全局模型和本地模型区分
        self.is_global = is_global  # 算法3.1第4-5行，区分全局模型和本地模型
        
        # 初始化全局权重
        self.global_weights = torch.ones((1, self.n_rf), dtype=torch.float32).to(self.device) / self.n_rf
        
    def create_hybrid_model(self, feature_extractor, regressor):
        """
        创建混合模型(本地特征提取器+全局回归器) (算法3.1第5行)
        
        Args:
            feature_extractor: 特征提取器
            regressor: 回归器
            
        Returns:
            hybrid_model: 混合模型
        """
        # 创建一个新的模型实例
        hybrid_model = FedGPAI_advanced(
            lam=self.lam,
            rf_feature=self.rf_feature,
            eta=self.eta,
            regressor_type=self.regressor_type,
            hidden_dims=self.hidden_dims,
            num_clients=self.num_clients,
            is_global=False  # 混合模型视为本地模型
        )
        
        # 设置特征提取器
        hybrid_model.feature_extractor = feature_extractor.clone().detach()
        
        # 根据回归器类型设置回归器
        try:
            if self.regressor_type == 'linear':
                if hasattr(regressor, 'weight'):
                    # 确保回归器已正确初始化具有相同维度
                    hybrid_model.regressor = regressor.__class__(regressor.weight.shape[0], regressor.weight.shape[1])
                    hybrid_model.regressor.weight.data.copy_(regressor.weight.data)
            else:  # MLP回归器
                # 对于MLP回归器，重新初始化整个模型并复制参数
                source_params = regressor.state_dict()
                hybrid_model.regressor.load_state_dict(source_params)
                
        except Exception as e:
            print(f"创建混合模型时出错: {str(e)}")
            # 如果初始化失败，尝试简单地共享同一回归器（不推荐，仅作为后备）
            hybrid_model.regressor = regressor
        
        return hybrid_model

    def extract_features(self, x):
        """
        提取特征
        
        Args:
            x: 输入数据，形状为 [batch_size, input_feature_dim]
            
        Returns:
            features: 提取的特征，形状为 [batch_size, n_rf]
        """
        # 将输入数据转换为PyTorch tensor并移动到正确设备
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32).to(self.device)
        elif x.device != self.device:
            x = x.to(self.device)
            
        # 检查并调整输入特征维度
        if x.shape[1] != self.input_feature_dim:
            with torch.no_grad():
                self.input_feature_dim = x.shape[1]
                # 重新初始化特征提取器以匹配输入维度
                self.feature_extractor = torch.randn(self.input_feature_dim, self.n_rf, dtype=torch.float32).to(self.device)
                self.feature_extractor.requires_grad_(True)
                # print(f"调整特征提取器维度: {self.feature_extractor.shape}")
        
        return torch.matmul(x, self.feature_extractor)
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入数据，形状为 [batch_size, input_feature_dim]
            
        Returns:
            output: 预测输出，形状为 [batch_size, 1]
        """
        # 提取特征
        features = self.extract_features(x)
        # 使用回归器预测
        return self.regressor(features)
    
    def fit(self, X, Y, batch_size=None, num_epochs=1, 
            regularizer=None, learning_rate=None, return_loss=False):
        """
        训练模型
        
        Args:
            X: 输入数据，形状为 [n_samples, input_feature_dim]
            Y: 目标数据，形状为 [n_samples, 1]
            batch_size: 批量大小
            num_epochs: 训练轮数
            regularizer: 正则化参数，如果为None则使用self.lam
            learning_rate: 学习率，如果为None则使用self.eta
            return_loss: 是否返回训练损失
            
        Returns:
            loss: 如果return_loss=True，返回每个epoch的训练损失
        """
        # 设置参数
        lam = regularizer if regularizer is not None else self.lam
        eta = learning_rate if learning_rate is not None else self.eta
        
        # 确保数据是张量
        if not torch.is_tensor(X):
            X = torch.tensor(X, dtype=torch.float32).to(self.device)
        if not torch.is_tensor(Y):
            Y = torch.tensor(Y, dtype=torch.float32).to(self.device)
        
        # 确保Y是二维的
        if len(Y.shape) == 1:
            Y = Y.reshape(-1, 1)
        
        # 计算批量大小
        if batch_size is None:
            batch_size = min(1000, X.shape[0])
        
        # 分批训练
        n_samples = X.shape[0]
        n_batches = max(1, n_samples // batch_size)
        
        # 记录损失
        losses = []
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            
            # 随机打乱数据
            indices = torch.randperm(n_samples)
            X_shuffled = X[indices]
            Y_shuffled = Y[indices]
            
            for i in range(n_batches):
                # 获取批量数据
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, n_samples)
                X_batch = X_shuffled[start_idx:end_idx]
                Y_batch = Y_shuffled[start_idx:end_idx]
                
                # 前向传播
                features = self.extract_features(X_batch)
                predictions = self.regressor(features)
                
                # 计算损失
                mse_loss = torch.mean((predictions - Y_batch) ** 2)
                
                # 添加正则化项
                # 对于特征提取器
                fe_reg_loss = lam * torch.sum(self.feature_extractor ** 2)
                
                # 对于回归器 - 根据不同类型处理
                reg_loss = 0.0
                if self.regressor_type == 'linear':
                    reg_loss = lam * torch.sum(self.regressor.weight ** 2)
                else:  # MLP
                    for param in self.regressor.parameters():
                        reg_loss += lam * torch.sum(param ** 2)
                
                total_loss = mse_loss + fe_reg_loss + reg_loss
                
                # 反向传播
                if self.feature_extractor.requires_grad:
                    grad_fe = torch.autograd.grad(total_loss, self.feature_extractor, retain_graph=True)[0]
                    self.feature_extractor.data -= eta * grad_fe
                
                # 更新回归器参数
                if self.regressor_type == 'linear':
                    if self.regressor.weight.requires_grad:
                        grad_reg = torch.autograd.grad(total_loss, self.regressor.weight)[0]
                        self.regressor.weight.data -= eta * grad_reg
                else:  # MLP
                    for param in self.regressor.parameters():
                        if param.requires_grad:
                            grad_param = torch.autograd.grad(total_loss, param, retain_graph=True)[0]
                            param.data -= eta * grad_param
                
                epoch_loss += mse_loss.item() * (end_idx - start_idx)
            
            # 计算平均损失
            epoch_loss /= n_samples
            losses.append(epoch_loss)
            
        if return_loss:
            return losses
    
    def predict(self, X):
        """
        使用模型进行预测
        
        Args:
            X: 输入数据，形状为 [n_samples, input_feature_dim]
            
        Returns:
            predictions: 预测结果，形状为 [n_samples, 1]
        """
        # 确保数据是张量
        if not torch.is_tensor(X):
            X = torch.tensor(X, dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            features = self.extract_features(X)
            predictions = self.regressor(features)
        
        # 返回PyTorch张量，而不是NumPy数组
        return predictions.detach()
    
    def get_global_model(self):
        """
        获取全局模型参数
        
        Returns:
            global_params: 全局模型参数
        """
        # 对于非线性回归器，返回参数字典
        if self.regressor_type != 'linear':
            return {name: param.clone() for name, param in self.regressor.named_parameters()}
        else:
            return self.regressor.weight.clone()
    
    def set_global_model(self, global_params):
        """
        设置全局模型参数
        
        Args:
            global_params: 全局模型参数
        """
        # 如果是字典，更新每个参数
        if isinstance(global_params, dict):
            for name, param in self.regressor.named_parameters():
                if name in global_params:
                    param.data.copy_(global_params[name])
        # 如果是单个张量，假设是线性回归器的权重
        elif isinstance(global_params, torch.Tensor):
            if self.regressor_type == 'linear':
                self.regressor.weight.data.copy_(global_params)
            else:
                raise ValueError("Expected a parameter dictionary for MLP regressor")
        else:
            raise TypeError("global_params must be a dict or a torch.Tensor")
            
    def evaluate_gradient_magnitude(self, global_model, local_model, X, Y):
        """
        评估梯度幅度 (算法3.2)
        
        Args:
            global_model: 全局模型
            local_model: 混合模型
            X: 输入数据
            Y: 目标值
            
        Returns:
            g_g: 全局模型梯度幅度
            g_i: 本地模型梯度幅度
        """
        # 使用torch.no_grad()避免计算图的积累
        with torch.no_grad():
            # 将输入转换为张量并确保在GPU上
            if not isinstance(X, torch.Tensor):
                X = torch.tensor(X, dtype=torch.float32).to(self.device)
            elif X.device != self.device:
                X = X.to(self.device)
                
            if not isinstance(Y, torch.Tensor):
                Y = torch.tensor(Y, dtype=torch.float32).to(self.device)
            elif Y.device != self.device:
                Y = Y.to(self.device)
            
            # 处理Y维度，确保其为列向量
            if len(Y.shape) == 0:  # 如果Y是单个标量
                Y = Y.reshape(1, 1)
            elif len(Y.shape) == 1:  # 如果Y是一维张量
                Y = Y.reshape(-1, 1)
            
            # 算法3.2第6-8行: 全局模型处理
            # 使用全局模型计算损失
            f_RF_global = global_model.predict(X)
            # 切断梯度
            if isinstance(f_RF_global, tuple):
                f_RF_global = f_RF_global[0]
            f_RF_global_detached = f_RF_global.detach()
            
            # 提取特征
            X_features_global = global_model.extract_features(X).detach()
            Y_detached = Y.detach()
            
            # 计算损失
            loss_diff_global = f_RF_global_detached - Y_detached
            
            # 手动计算梯度，避免依赖autograd
            X_global_t = X_features_global.t()
            grad_global = (2.0 / X.shape[0]) * torch.matmul(X_global_t, loss_diff_global)
            g_g = torch.norm(grad_global, p=2)
            
            # 释放中间变量
            del f_RF_global, f_RF_global_detached, loss_diff_global, X_global_t
            
            # 算法3.2第9-12行: 混合模型处理
            # 使用混合模型计算损失
            f_RF_local = local_model.predict(X)
            # 切断梯度
            if isinstance(f_RF_local, tuple):
                f_RF_local = f_RF_local[0]
            f_RF_local_detached = f_RF_local.detach()
            
            # 提取特征
            X_features_local = local_model.extract_features(X).detach()
            
            # 计算损失
            loss_diff_local = f_RF_local_detached - Y_detached
            
            # 手动计算梯度，避免依赖autograd
            X_local_t = X_features_local.t()
            grad_local = (2.0 / X.shape[0]) * torch.matmul(X_local_t, loss_diff_local)
            g_i = torch.norm(grad_local, p=2)
            
            # 释放中间变量
            del f_RF_local, f_RF_local_detached, X_features_local
            del loss_diff_local, X_local_t
            del X_features_global, Y_detached
        
        # 返回梯度幅度，已切断梯度图
        return g_g.detach(), g_i.detach()
        
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
        # 使用torch.no_grad()上下文管理器来避免计算图的积累
        with torch.no_grad():
            # 切断输入梯度并确保在GPU上
            g_g_detached = g_g.detach() if isinstance(g_g, torch.Tensor) else torch.tensor(g_g, dtype=torch.float32)
            g_i_detached = g_i.detach() if isinstance(g_i, torch.Tensor) else torch.tensor(g_i, dtype=torch.float32)
            
            # 移动到指定设备
            if g_g_detached.device != self.device:
                g_g_detached = g_g_detached.to(self.device)
            if g_i_detached.device != self.device:
                g_i_detached = g_i_detached.to(self.device)
            
            # 算法3.3第3-4行: 计算L2范数
            epsilon = 1e-8  # 防止除零
            L_g = torch.norm(g_g_detached, p=2)
            L_i = torch.norm(g_i_detached, p=2)
            
            # 计算插值参数
            alpha = L_i / (L_g + L_i + epsilon)
            
            # 处理不同类型的回归器
            if self.regressor_type == 'linear':
                # 线性回归器直接插值权重
                global_weight = global_regressor.weight.detach()
                local_weight = local_regressor.weight.detach()
                
                # 计算插值后的权重
                personalized_weight = alpha * global_weight + (1 - alpha) * local_weight
                
                # 创建新的线性回归器并设置权重
                new_regressor = personalized_weight.clone()
            else:  # MLP
                # 对于MLP回归器，需要逐层处理参数
                from copy import deepcopy
                new_regressor = deepcopy(global_regressor)  # 创建深拷贝避免影响原始回归器
                
                # 遍历全局和本地回归器的所有参数
                for (name, global_param), (_, local_param) in zip(
                    global_regressor.named_parameters(), local_regressor.named_parameters()):
                    # 获取当前层的全局和本地参数
                    global_param_detached = global_param.detach()
                    local_param_detached = local_param.detach()
                    
                    # 计算插值后的参数
                    personalized_param = alpha * global_param_detached + (1 - alpha) * local_param_detached
                    
                    # 为新回归器设置插值后的参数
                    for name_new, param_new in new_regressor.named_parameters():
                        if name_new == name:
                            param_new.data.copy_(personalized_param)
                            break
        
        return new_regressor
