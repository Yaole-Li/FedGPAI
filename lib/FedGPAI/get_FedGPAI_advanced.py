import torch
from lib.FedGPAI.FedGPAI_advanced import FedGPAI_advanced

def get_FedGPAI_advanced(model, args):
    """
    获取增强版FedGPAI模型实例（本地模型和全局模型），支持MLP特征提取器和MLP回归器
    
    Args:
        model: 数据特征维度信息
        args: 参数对象，需包含:
            - regressor_type: 回归器类型 ('linear'或'mlp')
            - hidden_dims: MLP回归器的隐藏层维度，如[64, 32]
            - extractor_hidden_dims: MLP特征提取器的隐藏层维度，如[128, 64]
            - output_dim: 特征提取器输出维度
        
    Returns:
        local: 本地模型列表
        federated: 联邦全局模型(只包含回归器)
        hybrid: 混合模型列表(本地特征提取器+全局回归器)
    """
    # 获取回归器相关参数
    regressor_type = getattr(args, 'regressor_type', 'mlp')
    hidden_dims = getattr(args, 'hidden_dims', [64, 32])
    
    # 获取特征提取器相关参数 (默认与回归器使用相同的隐藏层维度结构)
    extractor_hidden_dims = getattr(args, 'extractor_hidden_dims', hidden_dims)
    output_dim = getattr(args, 'output_dim', 100)  # 默认输出维度为100
    
    if args.task == "regression":
        # 创建全局联邦模型
        federated = FedGPAI_advanced(
            args.regularizer, 
            model, 
            args.eta, 
            regressor_type=regressor_type,
            regressor_hidden_dims=hidden_dims,
            extractor_hidden_dims=extractor_hidden_dims,
            output_dim=output_dim,
            num_clients=args.num_clients, 
            is_global=True
        )
        
        # 创建每个客户端的本地模型和混合模型
        local = []
        hybrid = []
        
        for i in range(args.num_clients):
            # 创建本地模型，包含本地特征提取器和本地回归器
            local_model = FedGPAI_advanced(
                args.regularizer, 
                model, 
                args.eta, 
                regressor_type=regressor_type,
                regressor_hidden_dims=hidden_dims,
                extractor_hidden_dims=extractor_hidden_dims,
                output_dim=output_dim,
                num_clients=args.num_clients, 
                is_global=False
            )
            local.append(local_model)
            
            # 创建混合模型 (本地特征提取器 + 全局回归器)
            hybrid_model = local_model.create_hybrid_model(
                local_model.feature_extractor, 
                federated.regressor
            )
            hybrid.append(hybrid_model)
    
    return local, federated, hybrid
