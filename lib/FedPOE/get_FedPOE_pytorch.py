import torch
from lib.FedPOE.FedPOE_regression_pytorch import FedPOE_regression

def get_FedPOE(model, args):
    """
    获取FedPOE模型的PyTorch实现
    
    参数:
        model: 随机特征或基础模型
        args: 参数配置
        
    返回:
        local: 本地模型列表
        federated: 全局联邦模型
    """
    if args.task == "classification":
        # 目前仅实现了回归模型，分类模型将在未来版本中添加
        raise NotImplementedError("PyTorch版本的FedPOE分类模型尚未实现")
    
    elif args.task == "regression":
        # 创建全局联邦模型
        federated = FedPOE_regression(args.regularizer, model, args.eta, args.num_clients)
        
        # 创建本地模型列表
        local = []
        for i in range(args.num_clients):
            local.append(FedPOE_regression(args.regularizer, model, args.eta, args.num_clients))
    
    return local, federated
