import torch
import numpy as np

class FedPOE_regression:
    """
    PyTorch实现的FedPOE回归算法
    """
    def __init__(self, lam, rf_feature, eta, num_clients):
        """
        初始化FedPOE回归模型
        
        参数:
            lam: 正则化参数
            rf_feature: 随机特征
            eta: 学习率
            num_clients: 客户端数量
        """
        self.lam = lam
        # 确保rf_feature是PyTorch张量
        if isinstance(rf_feature, np.ndarray):
            self.rf_feature = torch.tensor(rf_feature, dtype=torch.float32)
        else:
            self.rf_feature = rf_feature
            
        self.eta = eta
        # 初始化模型参数theta
        self.theta = torch.zeros((2 * rf_feature.shape[1], rf_feature.shape[2]), dtype=torch.float32)
        self.num_kernels = rf_feature.shape[2]
        self.num_clients = num_clients
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        
    def to(self, device):
        """
        将模型移到指定设备
        """
        self.device = device
        self.rf_feature = self.rf_feature.to(device)
        self.theta = self.theta.to(device)
        return self
        
    def predict(self, X, w):
        """
        使用模型进行预测
        
        参数:
            X: 输入特征
            w: 权重
            
        返回:
            f_RF: 组合预测
            f_RF_p: 各核函数的单独预测
            X_features: 提取的特征
        """
        # 确保输入是PyTorch张量
        if isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=torch.float32).to(self.device)
        if isinstance(w, np.ndarray):
            w = torch.tensor(w, dtype=torch.float32).to(self.device)
            
        M, N = X.shape
        a, n_components, b = self.rf_feature.shape
        X_f = torch.zeros((b, n_components), dtype=torch.float32).to(self.device)
        X_features = torch.zeros((b, 2 * n_components), dtype=torch.float32).to(self.device)
        f_RF_p = torch.zeros((b, 1), dtype=torch.float32).to(self.device)
        
        # 计算特征
        for j in range(b):
            X_f[j, :] = torch.matmul(X, self.rf_feature[:, :, j])
        
        # 连接sin和cos特征
        X_features = (1 / torch.sqrt(torch.tensor(n_components, dtype=torch.float32))) * torch.cat(
            (torch.sin(X_f), torch.cos(X_f)), dim=1
        )
        
        # 计算每个核的预测
        for j in range(b):
            f_RF_p[j, 0] = torch.matmul(X_features[j, :], self.theta[:, j])
        
        # 计算权重归一化并得到最终预测
        w_bar = w / torch.sum(w)
        f_RF = torch.matmul(w_bar, f_RF_p)
        
        return f_RF, f_RF_p, X_features
    
    def local_update(self, f_RF_p, Y, w, X_features):
        """
        执行本地更新
        
        参数:
            f_RF_p: 各核函数的单独预测
            Y: 真实标签
            w: 权重
            X_features: 提取的特征
            
        返回:
            w: 更新后的权重
            local_grad: 本地梯度
        """
        # 确保输入是PyTorch张量
        if isinstance(Y, np.ndarray):
            Y = torch.tensor(Y, dtype=torch.float32).to(self.device)
        elif isinstance(Y, (int, float)):
            Y = torch.tensor(Y, dtype=torch.float32).to(self.device)
            
        b, n_components = X_features.shape
        l = torch.zeros((1, self.num_kernels), dtype=torch.float32).to(self.device)
        local_grad = torch.zeros((n_components, self.num_kernels), dtype=torch.float32).to(self.device)
        
        # 计算每个核函数的损失和梯度
        for j in range(self.num_kernels):
            # 计算损失（带正则项）
            l[0, j] = (f_RF_p[j, 0] - Y)**2 + self.lam * (torch.norm(self.theta[:, j])**2)
            
            # 更新权重
            w[0, j] = w[0, j] * torch.exp(-self.eta * l[0, j])
            
            # 计算梯度
            local_grad[:, j] = self.eta * (
                (2 * (f_RF_p[j, 0] - Y) * X_features[j, :]) + 
                2 * self.lam * self.theta[:, j]
            )
            
        return w, local_grad
            
    def global_update(self, agg_grad):
        """
        执行全局更新
        
        参数:
            agg_grad: 聚合的梯度
        """
        theta_update = torch.zeros_like(self.theta).to(self.device)
        
        # 聚合梯度
        for i in range(len(agg_grad)):
            grad = agg_grad[i]
            if isinstance(grad, np.ndarray):
                grad = torch.tensor(grad, dtype=torch.float32).to(self.device)
                
            for j in range(self.num_kernels):
                theta_update[:, j] += (grad[:, j] / len(agg_grad))
        
        # 更新参数
        for i in range(self.num_kernels):
            self.theta[:, i] -= theta_update[:, i]
