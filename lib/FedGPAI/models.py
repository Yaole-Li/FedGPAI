import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# 检查是否有可用的GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class LinearRegressor(nn.Module):
    """
    线性回归器，与原始FedGPAI中的线性模型兼容
    """
    def __init__(self, input_dim, output_dim=1):
        super(LinearRegressor, self).__init__()
        self.weight = nn.Parameter(torch.zeros((input_dim, output_dim), dtype=torch.float32))
        self.to(device)
        
    def forward(self, x):
        # x的形状为 [batch_size, input_dim]
        return torch.matmul(x, self.weight)
    
    def get_parameters(self):
        """返回模型参数，用于联邦学习中的参数聚合"""
        return self.weight
    
    def set_parameters(self, new_params):
        """设置模型参数，用于联邦学习中的参数更新"""
        with torch.no_grad():
            self.weight.copy_(new_params)
    
    @staticmethod
    def interpolate(global_regressor, local_regressor, alpha=0.5):
        """线性插值两个模型的参数"""
        # 确保alpha在[0,1]之间
        alpha = max(0.0, min(1.0, alpha))
        with torch.no_grad():
            interpolated_weight = alpha * global_regressor.weight + (1 - alpha) * local_regressor.weight
        
        # 创建新的回归器
        new_regressor = LinearRegressor(global_regressor.weight.shape[0])
        new_regressor.set_parameters(interpolated_weight)
        return new_regressor


class MLPRegressor(nn.Module):
    """
    多层感知机(MLP)回归器，用于替代原始线性回归器
    """
    def __init__(self, input_dim, hidden_dims=[64, 32], output_dim=1, activation=nn.ReLU()):
        super(MLPRegressor, self).__init__()
        
        # 构建多层神经网络
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(activation)
            prev_dim = hidden_dim
            
        # 输出层
        layers.append(nn.Linear(prev_dim, output_dim))
        
        # 将所有层组合成序列模型
        self.model = nn.Sequential(*layers)
        self.to(device)
        
    def forward(self, x):
        # x的形状为 [batch_size, input_dim]
        return self.model(x)
    
    def get_parameters(self):
        """
        返回模型参数字典，用于联邦学习中的参数聚合
        """
        return {name: param for name, param in self.named_parameters()}
    
    def set_parameters(self, new_params):
        """
        设置模型参数，用于联邦学习中的参数更新
        new_params可以是字典或已排序的参数列表
        """
        with torch.no_grad():
            if isinstance(new_params, dict):
                # 如果是参数字典
                for name, param in self.named_parameters():
                    if name in new_params:
                        param.copy_(new_params[name])
            else:
                # 如果是单个参数张量（保持兼容性）
                # 尝试将参数设置到第一层
                first_layer = next(self.model.parameters(), None)
                if first_layer is not None and first_layer.shape == new_params.shape:
                    first_layer.copy_(new_params)
    
    @staticmethod
    def interpolate(global_regressor, local_regressor, alpha=0.5):
        """
        在全局MLP和本地MLP之间进行插值
        alpha: 插值因子，1表示完全使用全局模型，0表示完全使用本地模型
        """
        # 确保alpha在[0,1]之间
        alpha = max(0.0, min(1.0, alpha))
        
        # 获取模型结构信息
        if not isinstance(global_regressor, MLPRegressor) or not isinstance(local_regressor, MLPRegressor):
            raise TypeError("Both regressors must be MLPRegressor instances")
        
        # 创建新的回归器，结构与全局回归器相同
        input_dim = None
        hidden_dims = []
        
        # 假设模型是标准的线性层+激活函数结构
        for i, module in enumerate(global_regressor.model):
            if isinstance(module, nn.Linear):
                if input_dim is None:
                    input_dim = module.in_features
                if i < len(global_regressor.model) - 1:  # 不是最后一层
                    hidden_dims.append(module.out_features)
        
        # 创建新模型
        interpolated_regressor = MLPRegressor(input_dim, hidden_dims)
        
        # 插值参数
        with torch.no_grad():
            global_params = {name: param for name, param in global_regressor.named_parameters()}
            local_params = {name: param for name, param in local_regressor.named_parameters()}
            
            for name, param in interpolated_regressor.named_parameters():
                if name in global_params and name in local_params:
                    # 对相同名称的参数进行插值
                    interpolated_param = alpha * global_params[name] + (1 - alpha) * local_params[name]
                    param.copy_(interpolated_param)
        
        return interpolated_regressor


def create_regressor(regressor_type, input_dim, hidden_dims=None):
    """
    创建指定类型的回归器
    
    Args:
        regressor_type: 回归器类型，可选值为'linear', 'mlp'
        input_dim: 输入维度
        hidden_dims: MLP回归器的隐藏层维度列表
    
    Returns:
        回归器实例
    """
    if regressor_type == 'linear':
        return LinearRegressor(input_dim)
    elif regressor_type == 'mlp':
        if hidden_dims is None:
            # 默认MLP隐藏层结构
            hidden_dims = [64, 32]
        return MLPRegressor(input_dim, hidden_dims)
    else:
        raise ValueError(f"Unsupported regressor type: {regressor_type}")
