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
    def __init__(self, lam=1e-5, rf_feature=None, eta=1e-4, num_clients=1, is_global=False):
        """初始化FedGPAI回归模型
        
        参数:
            lam (float): 正则化参数
            rf_feature (numpy.ndarray 或 torch.Tensor): 随机特征张量
            eta (float): 学习率
            num_clients (int): 客户端数量
            is_global (bool): 是否为全局模型
        """
        # 设置设备，优先使用GPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.input_feature_dim = None
        
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
        self.regressor = torch.zeros((self.n_rf, 1), dtype=torch.float32, requires_grad=True).to(self.device)

        print(f"初始化模型 - 输入特征维度: {self.input_feature_dim}, RF特征维度: {self.n_rf}")
        print(f"特征提取器形状: {self.feature_extractor.shape}, 回归器形状: {self.regressor.shape}")
        
        # 初始化参数
        self.lam = lam  # 正则化参数
        self.eta = eta  # 学习率
        self.num_clients = num_clients  # 客户端数量
        
        # 全局模型和本地模型区分
        self.is_global = is_global  # 算法3.1第4-5行，区分全局模型和本地模型
        
        # 初始化全局和本地权重
        self.global_weights = torch.ones((1, self.n_rf), dtype=torch.float32).to(self.device) / self.n_rf
        
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
        # 如果特征在GPU上，首先分离梯度、复制到CPU再转换为NumPy数组
        rf_feature_np = global_feature_extractor.detach().cpu().numpy() if global_feature_extractor.is_cuda else global_feature_extractor.detach().numpy()
        
        # 创建混合模型
        hybrid_model = FedGPAI_regression(lam=self.lam, rf_feature=rf_feature_np, eta=self.eta, num_clients=self.num_clients, is_global=False)
        
        # 设置正确的维度参数
        hybrid_model.input_feature_dim = global_feature_extractor.shape[0]
        hybrid_model.n_rf = global_feature_extractor.shape[1]
        
        # 直接设置特征提取器和回归器
        hybrid_model.feature_extractor = global_feature_extractor.clone()
        hybrid_model.regressor = local_regressor.clone()
        
        # 确保维度匹配 - 回归器的行数应该等于特征提取器的列数
        if hybrid_model.feature_extractor.shape[1] != hybrid_model.regressor.shape[0]:
            print(f"警告：混合模型的特征提取器({hybrid_model.feature_extractor.shape})和回归器({hybrid_model.regressor.shape})维度不匹配")
            # 调整回归器维度以匹配特征提取器
            new_regressor = torch.zeros((hybrid_model.feature_extractor.shape[1], 1), dtype=torch.float32, requires_grad=True).to(self.device)
            # 复制可能的重叠部分
            min_dim = min(hybrid_model.regressor.shape[0], hybrid_model.feature_extractor.shape[1])
            new_regressor[:min_dim] = hybrid_model.regressor[:min_dim]
            hybrid_model.regressor = new_regressor
        
        # 确保混合模型的张量在正确的设备上
        if global_feature_extractor.device != self.device:
            hybrid_model.feature_extractor = hybrid_model.feature_extractor.to(self.device)
        
        if local_regressor.device != self.device:
            hybrid_model.regressor = hybrid_model.regressor.to(self.device)
            
        print(f"混合模型 - 特征提取器形状: {hybrid_model.feature_extractor.shape}, 回归器形状: {hybrid_model.regressor.shape}")
        
        return hybrid_model
        
    def extract_features(self, X):
        """提取特征
        参数:
            X (torch.Tensor): 输入数据
        
        返回:
            torch.Tensor: 提取的特征
        """
        # 将输入数据移至与特征提取器相同的设备
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32).to(self.device)
        elif X.device != self.device:
            X = X.to(self.device)
        
        # 确保特征提取器是梯度可跟踪的
        if not self.feature_extractor.requires_grad:
            self.feature_extractor.requires_grad_(True)
        
        # 检查并调整输入特征维度
        if X.shape[1] != self.input_feature_dim:
            self.input_feature_dim = X.shape[1]
            print(f"调整特征提取器维度以匹配输入维度 {X.shape[1]}")
            # 重新初始化特征提取器以匹配输入维度
            self.feature_extractor = torch.randn(self.input_feature_dim, self.n_rf, dtype=torch.float32).to(self.device)
            self.feature_extractor.requires_grad_(True)
        
        # 现在特征提取器应该是 [input_dim, n_rf]，而X是 [batch_size, input_dim]
        # 所以矩阵乘法应该是X @ feature_extractor，结果是 [batch_size, n_rf]
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
        
        # 检查输入维度，如果不匹配，可能需要调整特征提取器
        if X.shape[1] != self.input_feature_dim:
            print(f"预测时发现输入维度不匹配: 输入维度={X.shape[1]}, 特征提取器输入维度={self.input_feature_dim}")
        
        # 提取特征
        X_features = self.extract_features(X)
        
        # 验证特征维度与回归器维度是否匹配
        if X_features.shape[1] != self.regressor.shape[0]:
            print(f"警告：特征维度({X_features.shape[1]})与回归器输入维度({self.regressor.shape[0]})不匹配")
            # 如果需要，可以调整回归器以匹配特征
            new_regressor = torch.zeros((X_features.shape[1], 1), dtype=torch.float32, requires_grad=True).to(device)
            min_dim = min(self.regressor.shape[0], X_features.shape[1])
            new_regressor[:min_dim] = self.regressor[:min_dim]
            self.regressor = new_regressor
            print(f"已调整回归器维度为: {self.regressor.shape}")
        
        if w is not None:
            # 如果给定权重，则使用提供的权重
            f_RF_p = torch.zeros(X.shape[0], 1).to(device)
            for j in range(X.shape[0]):
                # 确保维度一致性的矩阵乘法
                f_RF_p[j, 0] = torch.matmul(X_features[j, :], self.regressor)
            return f_RF_p, f_RF_p, X_features
        else:
            # 否则使用默认模型参数
            try:
                f_RF_p = torch.matmul(X_features, self.regressor)
                return f_RF_p, f_RF_p, X_features
            except RuntimeError as e:
                print(f"矩阵乘法错误: {e}")
                print(f"X_features形状: {X_features.shape}, regressor形状: {self.regressor.shape}")
                # 尝试转置特征或回归器来匹配维度
                if X_features.shape[1] == self.regressor.shape[1] and self.regressor.shape[0] == 1:
                    # 回归器形状为[1, n]，需要转置为[n, 1]
                    self.regressor = self.regressor.t()
                    f_RF_p = torch.matmul(X_features, self.regressor)
                    return f_RF_p, f_RF_p, X_features
                raise
    
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
        # 重置所有梯度
        if hasattr(global_model, 'optimizer'):
            global_model.optimizer.zero_grad()
        if hasattr(local_model, 'optimizer'):
            local_model.optimizer.zero_grad()
            
        # 确保数据在GPU上
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32).to(device)
        elif X.device != device:
            X = X.to(device)
            
        if not isinstance(Y, torch.Tensor):
            Y = torch.tensor(Y, dtype=torch.float32).to(device)
        elif Y.device != device:
            Y = Y.to(device)
        
        # 处理Y维度，确保其为列向量
        if len(Y.shape) == 0:  # 如果Y是单个标量
            Y = Y.reshape(1, 1)
        elif len(Y.shape) == 1:  # 如果Y是一维张量
            Y = Y.reshape(-1, 1)
            
        # 确保模型参数需要梯度
        global_model.regressor.requires_grad_(True)
        local_model.regressor.requires_grad_(True)
        
        # 算法3.2第6-8行: 全局模型处理
        # 使用全局模型计算损失
        f_RF_global, _, X_features_global = global_model.predict(X)
        loss_global = torch.mean((f_RF_global - Y)**2)  # 平均均方误差损失
        
        # 手动计算梯度，避免依赖autograd
        X_global_t = X_features_global.t()
        grad_global = (2.0 / X.shape[0]) * torch.matmul(X_global_t, (f_RF_global - Y))
        g_g = torch.norm(grad_global, p=2)
        
        # 算法3.2第9-12行: 混合模型处理
        # 使用混合模型计算损失
        f_RF_local, _, X_features_local = local_model.predict(X)
        loss_local = torch.mean((f_RF_local - Y)**2)  # 平均均方误差损失
        
        # 手动计算梯度，避免依赖autograd
        X_local_t = X_features_local.t()
        grad_local = (2.0 / X.shape[0]) * torch.matmul(X_local_t, (f_RF_local - Y))
        g_i = torch.norm(grad_local, p=2)
        
        print(f"全局模型梯度范数: {g_g.item()}, 本地模型梯度范数: {g_i.item()}")
        
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
        # 如果不是全局模型，则不执行特征提取器更新
        if not self.is_global:
            return
            
        # 将聚合梯度转换为Torch张量并移到GPU
        if isinstance(agg_fe_grad, list):
            processed_grads = []
            for grad in agg_fe_grad:
                if isinstance(grad, torch.Tensor):
                    if grad.device != device:
                        processed_grads.append(grad.to(device))
                    else:
                        processed_grads.append(grad)
                else:
                    processed_grads.append(torch.tensor(grad, dtype=torch.float32).to(device))
            
            # 处理所有梯度并进行平均
            fe_update = torch.zeros_like(self.feature_extractor)
            for grad in processed_grads:
                # 确保梯度形状与特征提取器匹配
                if grad.shape == self.feature_extractor.shape:
                    fe_update += (grad / len(processed_grads))
                else:
                    print(f"\u8b66告: 特征提取器梯度形状 {grad.shape} 与特征提取器 {self.feature_extractor.shape} 不匹配")
                    # 尝试调整大小
                    try:
                        resized_grad = F.interpolate(grad.unsqueeze(0).unsqueeze(0), 
                                                     size=self.feature_extractor.shape, 
                                                     mode='nearest').squeeze(0).squeeze(0)
                        fe_update += (resized_grad / len(processed_grads))
                    except Exception as e:
                        print(f"调整梯度大小失败: {e}")
            
            # 更新特征提取器参数
            self.feature_extractor.data -= self.eta * fe_update
        else:
            print("错误: 特征提取器梯度必须是列表形式")
            return
        
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
        # 确保数据在GPU上
        if not isinstance(g_g, torch.Tensor):
            g_g = torch.tensor(g_g, dtype=torch.float32).to(device)
        elif g_g.device != device:
            g_g = g_g.to(device)
            
        if not isinstance(g_i, torch.Tensor):
            g_i = torch.tensor(g_i, dtype=torch.float32).to(device)
        elif g_i.device != device:
            g_i = g_i.to(device)
            
        if global_regressor.device != device:
            global_regressor = global_regressor.to(device)
            
        if local_regressor.device != device:
            local_regressor = local_regressor.to(device)
        
        # 算法3.3第3-4行: 计算L2范数
        L_g = torch.norm(g_g, p=2)  # 全局模型梯度幅度的L2范数
        L_i = torch.norm(g_i, p=2)  # 混合模型梯度幅度的L2范数
        
        # 创建用于存储个性化回归器参数的张量
        personalized_regressor = torch.zeros_like(global_regressor).to(device)
        
        # 遍历处理回归器各项参数
        epsilon = 1e-8  # 防止除零
        
        # 获取参数数量
        param_count = global_regressor.numel()
        
        # 将回归器展平为一维向量方便遍历
        global_flat = global_regressor.view(-1)
        local_flat = local_regressor.view(-1)
        pers_flat = personalized_regressor.view(-1)
        
        # 算法3.3第5-12行: 逐参数遍历
        for j in range(param_count):
            # 算法3.3第7-8行: 归一化处理
            g_g_j_norm = g_g.view(-1)[j % g_g.numel()] / (L_g + epsilon)
            g_i_j_norm = g_i.view(-1)[j % g_i.numel()] / (L_i + epsilon)
            
            # 算法3.3第10行: 计算插值权重
            alpha_j = 1.0 - g_i_j_norm / (g_i_j_norm + g_g_j_norm + epsilon)
            
            # 算法3.3第12行: 调整融合比例
            pers_flat[j] = alpha_j * local_flat[j] + (1.0 - alpha_j) * global_flat[j]
        
        # 恢复回归器原始形状
        personalized_regressor = pers_flat.view_as(global_regressor)
        
        return personalized_regressor
