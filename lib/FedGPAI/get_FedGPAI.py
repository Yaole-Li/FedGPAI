import torch
from lib.FedGPAI.FedGPAI_regression import FedGPAI_regression

def get_FedGPAI(model, args):
    """
    获取FedGPAI模型实例（本地模型和全局模型），根据算法3.1
    
    Args:
        model: 随机特征模型
        args: 参数对象
        
    Returns:
        local: 本地模型列表
        federated: 联邦全局模型
        hybrid: 混合模型列表(全局特征提取器+本地回归器)
    """
    if args.task == "regression":
        # 创建全局联邦模型 (算法3.1第1-2行)
        federated = FedGPAI_regression(args.regularizer, model, args.eta, args.num_clients, is_global=True)
        
        # 创建每个客户端的本地模型和混合模型 (算法3.1第3-5行)
        local = []
        hybrid = []
        
        for i in range(args.num_clients):
            # 创建本地模型
            local_model = FedGPAI_regression(args.regularizer, model, args.eta, args.num_clients, is_global=False)
            local.append(local_model)
            
            # 创建混合模型 (全局特征提取器 + 本地回归器)
            # 这是算法3.1第5行: θ_i^t ← (ω^t, φ_i^{t-1})
            # 在初始状态下，先使用与本地模型相同的参数
            hybrid_model = local_model.create_hybrid_model(federated.feature_extractor, local_model.regressor)
            hybrid.append(hybrid_model)
    
    return local, federated, hybrid
