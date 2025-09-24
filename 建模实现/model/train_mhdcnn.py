#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MHDCNN模型训练启动脚本
简化版本，便于快速开始训练
"""

import sys
import os
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
import logging

# 添加当前目录到Python路径
sys.path.append(str(Path(__file__).parent))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_environment():
    """检查运行环境"""
    logger.info("检查运行环境...")
    
    # 检查PyTorch
    logger.info(f"PyTorch版本: {torch.__version__}")
    
    # 检查CUDA
    if torch.cuda.is_available():
        logger.info(f"CUDA可用，设备数: {torch.cuda.device_count()}")
        logger.info(f"当前CUDA设备: {torch.cuda.get_device_name(0)}")
    else:
        logger.info("CUDA不可用，将使用CPU")
    
    # 检查数据目录
    data_dir = Path("/Users/gsyoung/MBP_documents/code/Matlab/研究生数学建模/建模实现/处理后数据/dataset")
    if data_dir.exists():
        logger.info(f"数据目录存在: {data_dir}")
        fold_dirs = list(data_dir.glob("fold_*"))
        logger.info(f"找到 {len(fold_dirs)} 个fold目录")
        
        # 检查第一个fold的样本
        if fold_dirs:
            sample_dirs = list(fold_dirs[0].iterdir())
            logger.info(f"Fold 1 包含 {len(sample_dirs)} 个样本")
            
            if sample_dirs:
                sample_dir = sample_dirs[0]
                sample_files = list(sample_dir.glob("*"))
                logger.info(f"样本文件: {[f.name for f in sample_files]}")
    else:
        logger.error(f"数据目录不存在: {data_dir}")
        return False
    
    return True

def quick_data_test():
    """快速数据加载测试"""
    logger.info("进行快速数据加载测试...")
    
    try:
        from mhdcnn_model import BearingDataset
        
        data_dir = "/Users/gsyoung/MBP_documents/code/Matlab/研究生数学建模/建模实现/处理后数据/dataset"
        
        # 创建测试数据集
        test_dataset = BearingDataset(data_dir, fold_num=1)
        logger.info(f"数据集大小: {len(test_dataset)}")
        
        # 加载第一个样本
        sample = test_dataset[0]
        logger.info(f"样本结构:")
        logger.info(f"  图像形状: {sample['image'].shape}")
        logger.info(f"  序列形状: {sample['sequence'].shape}")
        logger.info(f"  CSV特征形状: {sample['csv_features'].shape}")
        logger.info(f"  标签: {sample['label'].item()}")
        logger.info(f"  文件夹名: {sample['folder_name']}")
        
        return True
        
    except Exception as e:
        logger.error(f"数据加载测试失败: {e}")
        return False

def quick_model_test():
    """快速模型创建测试"""
    logger.info("进行快速模型创建测试...")
    
    try:
        from mhdcnn_model import MHDCNN
        
        # 创建模型
        model = MHDCNN(csv_input_dim=50, num_classes=4)
        
        # 计算参数数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        logger.info(f"模型创建成功")
        logger.info(f"总参数数: {total_params:,}")
        logger.info(f"可训练参数数: {trainable_params:,}")
        
        # 测试前向传播
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        
        # 创建测试输入
        batch_size = 2
        test_images = torch.randn(batch_size, 3, 224, 224).to(device)
        test_sequences = torch.randn(batch_size, 20000).to(device)
        test_csv = torch.randn(batch_size, 50).to(device)
        
        # 前向传播
        model.eval()
        with torch.no_grad():
            output = model(test_images, test_sequences, test_csv)
            logger.info(f"模型输出形状: {output.shape}")
            logger.info(f"输出范围: [{output.min().item():.4f}, {output.max().item():.4f}]")
        
        return True
        
    except Exception as e:
        logger.error(f"模型测试失败: {e}")
        return False

def start_training(quick_test=True):
    """开始训练"""
    logger.info("准备开始MHDCNN模型训练...")
    
    if quick_test:
        logger.info("执行快速测试模式（减少epoch数）")
        
    try:
        from mhdcnn_model import k_fold_cross_validation
        
        data_dir = "/Users/gsyoung/MBP_documents/code/Matlab/研究生数学建模/建模实现/处理后数据/dataset"
        
        # 根据是否快速测试调整参数
        num_epochs = 5 if quick_test else 50
        num_folds = 3 if quick_test else 10
        
        logger.info(f"开始K折交叉验证训练...")
        logger.info(f"训练参数: {num_folds}折交叉验证, {num_epochs}个epoch")
        
        # 开始训练
        fold_results, avg_accuracy, std_accuracy = k_fold_cross_validation(
            data_dir=data_dir,
            num_folds=num_folds,
            num_epochs=num_epochs
        )
        
        logger.info(f"训练完成!")
        logger.info(f"平均准确率: {avg_accuracy:.4f} ± {std_accuracy:.4f}")
        
        return True
        
    except Exception as e:
        logger.error(f"训练失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def main():
    """主函数"""
    logger.info("=" * 60)
    logger.info("MHDCNN轴承故障诊断模型训练")
    logger.info("=" * 60)
    
    # 1. 检查环境
    if not check_environment():
        logger.error("环境检查失败，退出程序")
        return
    
    # 2. 数据测试
    if not quick_data_test():
        logger.error("数据加载测试失败，请检查数据格式")
        return
    
    # 3. 模型测试
    if not quick_model_test():
        logger.error("模型创建测试失败，请检查模型定义")
        return
    
    # 4. 询问是否开始训练
    print("\n" + "=" * 60)
    print("环境检查和模型测试完成!")
    print("=" * 60)
    
    choice = input("是否开始训练? \n1. 快速测试训练(3折,5epoch)\n2. 完整训练(10折,50epoch)\n3. 退出\n请选择 (1/2/3): ").strip()
    
    if choice == '1':
        logger.info("开始快速测试训练...")
        start_training(quick_test=True)
    elif choice == '2':
        logger.info("开始完整训练...")
        start_training(quick_test=False)
    else:
        logger.info("退出程序")
        return
    
    logger.info("=" * 60)
    logger.info("程序执行完成")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()
