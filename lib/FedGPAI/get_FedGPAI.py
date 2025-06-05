import torch
from lib.FedGPAI.FedGPAI_regression import FedGPAI_regression

def get_FedGPAI(model, args):
    """
    获取FedGPAI模型实例（本地模型和全局模型），根据算法3.1改进版
    
    Args:
        model: 随机特征模型
        args: 参数对象
        
    Returns:
        local: 本地模型列表
        federated: 联邦全局模型(只包含回归器)
        hybrid: 混合模型列表(本地特征提取器+全局回归器)
    """
    if args.task == "regression":
        # 创建全局联邦模型 (算法3.1第1-2行)
        # 全局模型只使用回归器部分
        federated = FedGPAI_regression(args.regularizer, model, args.eta, args.num_clients, is_global=True)
        
        # 创建每个客户端的本地模型和混合模型 (算法3.1第3-5行)
        local = []
        hybrid = []
        
        for i in range(args.num_clients):
            # 创建本地模型，包含本地特征提取器和本地回归器
            local_model = FedGPAI_regression(args.regularizer, model, args.eta, args.num_clients, is_global=False)
            local.append(local_model)
            
            # 创建混合模型 (本地特征提取器 + 全局回归器)
            # 这是算法3.1改进版第4-5行: θ^t ← (ω_i^t, φ^t) 和 θ_i^t ← (ω_i^t, φ_i^{t-1})
            hybrid_model = local_model.create_hybrid_model(local_model.feature_extractor, federated.regressor)
            hybrid.append(hybrid_model)
    
    return local, federated, hybrid
