import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, List
from copy import deepcopy
from lib.FedGPAI.models import create_regressor, LinearRegressor, MLPRegressor, MLPFeatureExtractor

# 检查是否有可用的GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class FedGPAI_advanced:
    """
    增强版FedGPAI模型类，支持可选的回归器类型(线性或MLP)
    实现算法3.1-3.3的增强版本，特征提取器和回归器分离的联邦学习
    """
    def __init__(self, lam=1e-5, rf_feature=None, eta=1e-4, regressor_type='mlp', 
                 extractor_hidden_dims=None, regressor_hidden_dims=None, output_dim=64, num_clients=1, is_global=False):
        """初始化增强版FedGPAI模型
        
        参数:
            lam (float): 正则化参数
            rf_feature (numpy.ndarray 或 torch.Tensor): 数据特征（仅用于维度推断，不再用于实际特征提取）
            eta (float): 学习率
            regressor_type (str): 回归器类型，默认为'mlp'
            extractor_hidden_dims (List[int]): 特征提取器MLP的隐藏层大小，None时使用默认值
            regressor_hidden_dims (List[int]): 回归器MLP的隐藏层大小，None时使用默认值
            output_dim (int): MLP特征提取器的输出维度，也是回归器的输入维度
            num_clients (int): 客户端数量
            is_global (bool): 是否为全局模型
        """
        # 设置设备，优先使用GPU
        self.device = device
        self.input_feature_dim = None
        self.regressor_type = regressor_type
        
        # 设置MLP隐藏层大小
        self.extractor_hidden_dims = extractor_hidden_dims if extractor_hidden_dims else [64, 32]
        self.regressor_hidden_dims = regressor_hidden_dims if regressor_hidden_dims else [64, 32]
        
        # 保存输出维度
        self.output_dim = output_dim
        
        # 处理不同类型的rf_feature参数
        if isinstance(rf_feature, int):
            # 如果rf_feature是整数，就直接将其作为输入维度
            self.input_feature_dim = rf_feature
            self.rf_feature = None
        elif isinstance(rf_feature, np.ndarray):
            # 如果是numpy数组，转换为torch张量
            self.rf_feature = torch.from_numpy(rf_feature).float().to(self.device)
            if len(rf_feature.shape) == 3:
                n_samples, n_features, n_rf = rf_feature.shape
                self.input_feature_dim = n_features
            elif len(rf_feature.shape) > 1:
                self.input_feature_dim = rf_feature.shape[1] 
            else:
                self.input_feature_dim = 32
        elif isinstance(rf_feature, torch.Tensor):
            # 如果是torch张量
            self.rf_feature = rf_feature
            if len(rf_feature.shape) == 3:
                n_samples, n_features, n_rf = rf_feature.shape
                self.input_feature_dim = n_features
            elif len(rf_feature.shape) > 1:
                self.input_feature_dim = rf_feature.shape[1]
            else:
                self.input_feature_dim = 32
        else:
            # 默认输入维度
            self.input_feature_dim = 32
            self.rf_feature = None
            
        # 使用output_dim作为特征提取器的输出维度
        self.feature_dim = output_dim
        
        # 初始化MLP特征提取器
        self.feature_extractor = MLPFeatureExtractor(
            input_dim=self.input_feature_dim,
            output_dim=self.feature_dim,
            hidden_dims=self.extractor_hidden_dims
        )
        
        # 初始化MLP回归器
        self.regressor = MLPRegressor(
            input_dim=self.feature_dim,
            hidden_dims=self.regressor_hidden_dims,
            output_dim=1
        )
        
        # 初始化参数
        self.lam = lam  # 正则化参数
        self.eta = eta  # 学习率
        self.num_clients = num_clients  # 客户端数量
        
        # 全局模型和本地模型区分
        self.is_global = is_global  # 算法3.1第4-5行，区分全局模型和本地模型
        
        # 初始化全局权重 (适配MLP特征提取器，不再使用随机傲里叶特征)
        # 对于MLP特征提取器，我们只保留这个属性但不实际使用
        self.feature_weights = torch.ones((1, self.output_dim), dtype=torch.float32).to(self.device) / self.output_dim
        
    def create_hybrid_model(self, feature_extractor, regressor):
        """
        创建混合模型(本地特征提取器+全局回归器) (算法3.1第5行)
        
        Args:
            feature_extractor: 本地MLP特征提取器
            regressor: 全局MLP回归器
            
        Returns:
            hybrid_model: 混合模型
        """
        # 创建一个新的模型实例
        hybrid_model = FedGPAI_advanced(
            lam=self.lam,
            rf_feature=self.input_feature_dim,  # 直接传递输入维度作为整数
            eta=self.eta,
            regressor_type=self.regressor_type,
            extractor_hidden_dims=self.extractor_hidden_dims,
            regressor_hidden_dims=self.regressor_hidden_dims,
            output_dim=self.output_dim,  # 确保传递output_dim参数
            num_clients=self.num_clients,
            is_global=False  # 混合模型视为本地模型
        )
        
        # 设置特征提取器，使用深复制而不是简单的clone
        try:
            hybrid_model.feature_extractor = deepcopy(feature_extractor)
            hybrid_model.input_feature_dim = feature_extractor.model[0].in_features
            hybrid_model.feature_dim = feature_extractor.model[-1].out_features
        except Exception as e:
            print(f"复制本地特征提取器时出错: {str(e)}")
        
        # 设置回归器（使用全局回归器）
        try:
            # 对于MLP回归器，重新初始化并复制参数
            hybrid_model.regressor = deepcopy(regressor)
        except Exception as e:
            print(f"复制全局回归器时出错: {str(e)}")
            # 如果复制失败，创建新的回归器并加载状态
            if hasattr(regressor, 'state_dict'):
                source_params = regressor.state_dict()
                hybrid_model.regressor = MLPRegressor(
                    input_dim=hybrid_model.feature_dim,
                    hidden_dims=self.regressor_hidden_dims,
                    output_dim=1
                )
                hybrid_model.regressor.load_state_dict(source_params)
        
        return hybrid_model

    def extract_features(self, x):
        """
        使用MLP特征提取器进行特征提取
        
        Args:
            x: 输入数据，形状为 [batch_size, input_feature_dim]
            
        Returns:
            features: 提取的特征，形状为 [batch_size, feature_dim]
        """
        # 将输入数据转换为PyTorch tensor并移动到正确设备
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32).to(self.device)
        elif x.device != self.device:
            x = x.to(self.device)
            
        # 检查输入维度是否与提取器的输入层一致
        if x.shape[1] != self.input_feature_dim:
            print(f"Warning: 输入特征维度不匹配，需要 {self.input_feature_dim} 实际为 {x.shape[1]}")
            # 重新初始化MLP特征提取器以匹配新的输入维度
            self.input_feature_dim = x.shape[1]
            self.feature_extractor = MLPFeatureExtractor(
                input_dim=self.input_feature_dim,
                output_dim=self.feature_dim,
                hidden_dims=self.extractor_hidden_dims
            )
        
        # 使用MLP特征提取器进行前向传播
        return self.feature_extractor(x)
    
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
                # 对于MLP特征提取器，需要对所有参数进行正则化
                fe_reg_loss = 0.0
                for param in self.feature_extractor.parameters():
                    fe_reg_loss += lam * torch.sum(param ** 2)
                
                # 对于回归器 - 根据不同类型处理
                reg_loss = 0.0
                if self.regressor_type == 'linear':
                    reg_loss = lam * torch.sum(self.regressor.weight ** 2)
                else:  # MLP
                    for param in self.regressor.parameters():
                        reg_loss += lam * torch.sum(param ** 2)
                
                total_loss = mse_loss + fe_reg_loss + reg_loss
                
                # 反向传播 - 对MLP特征提取器的每个参数分别更新
                for param in self.feature_extractor.parameters():
                    if param.requires_grad:
                        grad_param = torch.autograd.grad(total_loss, param, retain_graph=True)[0]
                        param.data -= eta * grad_param
                
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
        
    def model_interpolation(self, global_regressor, local_regressor, L_g, L_i, epsilon=1e-8):
        """
        模型参数插值方法 - 实现算法3.3
        
        Args:
            global_regressor: 全局MLP回归器
            local_regressor: 本地MLP回归器
            L_g: 全局模型梯度大小
            L_i: 本地模型梯度大小
            epsilon: 防止除零的小量
            
        Returns:
            personalized_regressor: 个性化插值回归器
        """
        with torch.no_grad():
            # 创建新的MLP回归器，通过深复制全局回归器
            new_regressor = deepcopy(global_regressor)
            
            # 逐参数插值处理，与算法3.3一致
            for (name, global_param), (_, local_param) in zip(
                global_regressor.named_parameters(), local_regressor.named_parameters()
            ):
                # 对单个参数计算其归一化梯度
                # 梯度幅度归一化处理
                g_g_norm = torch.norm(global_param.detach()) / (L_g + epsilon)
                g_i_norm = torch.norm(local_param.detach()) / (L_i + epsilon)
                
                # 按算法3.3公式计算插值权重
                alpha_j = 1.0 - (g_i_norm / (g_i_norm + g_g_norm + epsilon))
                
                # 插值计算
                personalized_param = alpha_j * global_param.detach() + (1.0 - alpha_j) * local_param.detach()
                
                # 获取new_regressor中对应参数并设置插值后的值
                param_new = dict(new_regressor.named_parameters())[name]
                param_new.data.copy_(personalized_param)
            
            # 返回插值后的新回归器
            return new_regressor
