import numpy as np
from lib.datasets.air.data_processor import data_processing_air
from lib.datasets.wec.data_processor import data_processing_wec
from lib.datasets.water.data_processor import data_processing_water
from lib.datasets.product.data_processor import data_processing_product
from lib.datasets.power.data_processor import data_processing_power

def data_loader(args):
    """
    根据参数加载指定的数据集
    
    Args:
        args: 包含数据集名称和其他配置参数的对象

    Returns:
        X, Y: 特征和标签数据，以字典形式组织，每个客户端对应一个键
    """
    if args.dataset=="Air":
        X, Y = data_processing_air(args)
    
    elif args.dataset=="WEC":
        X, Y = data_processing_wec(args)

    elif args.dataset=="Water":
        X, Y = data_processing_water(args)
    
    elif args.dataset=="Product":
        X, Y = data_processing_product(args)
    
    elif args.dataset=="Power":
        X, Y = data_processing_power(args)
    
    return X, Y
