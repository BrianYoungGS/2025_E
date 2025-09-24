#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
轴承故障诊断数据加载示例
演示如何使用data_loader和data_preprocessor模块
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# 添加模块路径
sys.path.append(str(Path(__file__).parent))

from data_loader import BearingDataLoader, create_train_test_split
from data_preprocessor import BearingDataPreprocessor, create_preprocessing_pipeline

def main():
    """
    主函数：演示完整的数据加载和预处理流程
    """
    print("=" * 60)
    print("轴承故障诊断数据加载和预处理示例")
    print("=" * 60)
    
    # 1. 创建数据加载器
    print("\n📁 创建数据加载器...")
    loader = BearingDataLoader()
    
    # 2. 扫描数据集
    print("🔍 扫描数据集...")
    segment_info, raw_info = loader.scan_datasets()
    
    # 3. 获取数据集统计信息
    print("\n📊 数据集统计信息:")
    stats = loader.get_dataset_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # 4. 筛选训练数据
    print("\n🎯 筛选训练数据...")
    
    # 方案1: 使用分段数据，12kHz，驱动端
    train_filter_config = {
        'fault_types': ['N', 'B', 'IR', 'OR'],  # 四种故障类型
        'sampling_freqs': ['12kHz'],             # 12kHz数据
        'sensor_types': ['DE'],                  # 驱动端数据
        'data_type': 'segment',                  # 分段数据
        'loads': [0, 1, 2, 3]                   # 所有载荷条件
    }
    
    filtered_data = loader.filter_by_criteria(**train_filter_config)
    print(f"筛选结果: {len(filtered_data)} 个样本")
    
    # 显示每个故障类型的样本数量
    fault_counts = {}
    for data in filtered_data:
        fault_type = data['fault_type']
        fault_counts[fault_type] = fault_counts.get(fault_type, 0) + 1
    print(f"故障类型分布: {fault_counts}")
    
    # 5. 创建训练/测试分割
    print("\n📈 创建训练/测试数据集...")
    
    # 限制样本数量以快速演示（实际使用时可以移除这个限制）
    demo_data = filtered_data[:100]  # 仅使用前100个样本进行演示
    
    # 加载数据
    dataset = loader.load_dataset(
        demo_data,
        include_raw=True,           # 包含原始信号
        include_features=True,      # 包含特征数据
        include_freq_analysis=False, # 不包含频域分析（可选）
        include_images=False        # 不包含图像（加快速度）
    )
    
    print(f"加载的数据形状:")
    print(f"  原始信号: {dataset['raw_signals'].shape}")
    print(f"  特征数据: {dataset['features'].shape}")
    print(f"  标签: {dataset['labels'].shape}")
    
    # 6. 数据预处理
    print("\n🔧 数据预处理...")
    preprocessor = BearingDataPreprocessor()
    
    # 原始信号预处理
    if len(dataset['raw_signals']) > 0:
        # 标准化原始信号
        normalized_signals = preprocessor.normalize_signals(
            dataset['raw_signals'], 
            method='standard'
        )
        print(f"信号标准化完成: {normalized_signals.shape}")
        
        # 提取综合特征
        extracted_features = preprocessor.extract_comprehensive_features(
            normalized_signals,
            fs=12000,
            include_wavelet=False  # 跳过小波特征以加快演示
        )
        print(f"特征提取完成: {extracted_features.shape}")
        
        # 特征选择
        if len(extracted_features) > 0:
            selected_features = preprocessor.feature_selection(
                extracted_features,
                dataset['labels'],
                method='univariate',
                k=min(30, extracted_features.shape[1])  # 选择30个最佳特征
            )
            print(f"特征选择完成: {selected_features.shape}")
    
    # 7. 数据增强示例
    print("\n🚀 数据增强示例...")
    if len(dataset['raw_signals']) > 0:
        # 仅对少量数据进行增强演示
        sample_signals = dataset['raw_signals'][:10]
        sample_labels = dataset['labels'][:10]
        
        augmented_signals, augmented_labels = preprocessor.augment_signals(
            sample_signals,
            sample_labels,
            methods=['noise', 'scaling'],
            noise_level=0.005
        )
        print(f"数据增强: {sample_signals.shape[0]} -> {augmented_signals.shape[0]} 个样本")
    
    # 8. 数据可视化
    print("\n📊 数据可视化...")
    if len(dataset['raw_signals']) > 0:
        plot_data_examples(dataset, fault_counts)
    
    # 9. 迁移学习数据准备
    print("\n🎯 迁移学习数据准备...")
    transfer_learning_demo(loader)
    
    print("\n✅ 数据加载和预处理示例完成！")
    print("=" * 60)

def plot_data_examples(dataset, fault_counts):
    """
    绘制数据示例
    """
    try:
        import matplotlib.pyplot as plt
        
        # 创建图形
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('轴承故障信号示例', fontsize=16)
        
        # 故障类型标签映射
        fault_labels = {0: 'Normal', 1: 'Ball', 2: 'Inner Race', 3: 'Outer Race'}
        
        # 为每种故障类型绘制一个示例
        plot_idx = 0
        for fault_type in [0, 1, 2, 3]:  # N, B, IR, OR
            if fault_type in dataset['labels'] and plot_idx < 4:
                # 找到该故障类型的第一个样本
                indices = np.where(dataset['labels'] == fault_type)[0]
                if len(indices) > 0:
                    signal = dataset['raw_signals'][indices[0]]
                    
                    ax = axes[plot_idx // 2, plot_idx % 2]
                    ax.plot(signal[:1000])  # 只显示前1000个点
                    ax.set_title(f'{fault_labels[fault_type]} 故障')
                    ax.set_xlabel('采样点')
                    ax.set_ylabel('幅值')
                    ax.grid(True, alpha=0.3)
                    
                    plot_idx += 1
        
        plt.tight_layout()
        
        # 保存图像
        output_dir = Path(__file__).parent.parent / "outputs"
        output_dir.mkdir(exist_ok=True)
        plt.savefig(output_dir / "bearing_signal_examples.png", dpi=150, bbox_inches='tight')
        print(f"  信号示例图保存至: {output_dir / 'bearing_signal_examples.png'}")
        
        # 绘制故障类型分布
        plt.figure(figsize=(8, 6))
        fault_types = list(fault_counts.keys())
        fault_nums = list(fault_counts.values())
        
        plt.bar(fault_types, fault_nums, color=['green', 'orange', 'red', 'blue'])
        plt.title('故障类型分布')
        plt.xlabel('故障类型')
        plt.ylabel('样本数量')
        plt.grid(True, alpha=0.3)
        
        for i, v in enumerate(fault_nums):
            plt.text(i, v + 1, str(v), ha='center', va='bottom')
        
        plt.savefig(output_dir / "fault_distribution.png", dpi=150, bbox_inches='tight')
        print(f"  故障分布图保存至: {output_dir / 'fault_distribution.png'}")
        
        plt.close('all')  # 关闭所有图形
        
    except ImportError:
        print("  matplotlib未安装，跳过可视化")
    except Exception as e:
        print(f"  可视化过程中出现错误: {e}")

def transfer_learning_demo(loader):
    """
    迁移学习数据准备演示
    """
    print("\n  为迁移学习准备不同域的数据...")
    
    # 源域：12kHz DE数据
    source_data = loader.filter_by_criteria(
        fault_types=['N', 'B', 'IR', 'OR'],
        sampling_freqs=['12kHz'],
        sensor_types=['DE'],
        data_type='segment'
    )
    
    # 目标域：12kHz FE数据（模拟不同传感器位置）
    target_data = loader.filter_by_criteria(
        fault_types=['N', 'B', 'IR', 'OR'],
        sampling_freqs=['12kHz'],
        sensor_types=['FE'],
        data_type='segment'
    )
    
    # 另一个目标域：48kHz数据（模拟不同采样频率）
    target_data_48k = loader.filter_by_criteria(
        fault_types=['N', 'B', 'IR', 'OR'],
        sampling_freqs=['48kHz'],
        data_type='raw'
    )
    
    print(f"  源域数据 (12kHz DE): {len(source_data)} 个样本")
    print(f"  目标域1 (12kHz FE): {len(target_data)} 个样本")
    print(f"  目标域2 (48kHz): {len(target_data_48k)} 个样本")
    
    # 分析域差异
    analyze_domain_differences(source_data, target_data, "12kHz DE vs FE")
    
    return source_data, target_data, target_data_48k

def analyze_domain_differences(source_data, target_data, comparison_name):
    """
    分析不同域之间的数据分布差异
    """
    print(f"\n  分析域差异: {comparison_name}")
    
    # 故障类型分布对比
    source_faults = {}
    target_faults = {}
    
    for data in source_data:
        fault = data['fault_type']
        source_faults[fault] = source_faults.get(fault, 0) + 1
    
    for data in target_data:
        fault = data['fault_type']
        target_faults[fault] = target_faults.get(fault, 0) + 1
    
    print(f"    源域故障分布: {source_faults}")
    print(f"    目标域故障分布: {target_faults}")
    
    # 载荷分布对比
    source_loads = {}
    target_loads = {}
    
    for data in source_data:
        load = data['load']
        source_loads[load] = source_loads.get(load, 0) + 1
    
    for data in target_data:
        load = data['load']
        target_loads[load] = target_loads.get(load, 0) + 1
    
    print(f"    源域载荷分布: {source_loads}")
    print(f"    目标域载荷分布: {target_loads}")

def create_benchmark_dataset():
    """
    创建基准数据集配置
    """
    print("\n🎯 创建基准数据集配置...")
    
    # 不同的实验配置
    configs = {
        'config_1': {
            'name': '12kHz驱动端全数据',
            'filter': {
                'sampling_freqs': ['12kHz'],
                'sensor_types': ['DE'],
                'data_type': 'segment'
            }
        },
        'config_2': {
            'name': '12kHz风扇端全数据',
            'filter': {
                'sampling_freqs': ['12kHz'],
                'sensor_types': ['FE'],
                'data_type': 'segment'
            }
        },
        'config_3': {
            'name': '48kHz全数据',
            'filter': {
                'sampling_freqs': ['48kHz'],
                'data_type': 'raw'
            }
        },
        'config_4': {
            'name': '混合域数据',
            'filter': {
                'data_type': 'both'
            }
        }
    }
    
    for config_name, config in configs.items():
        print(f"  {config_name}: {config['name']}")
        print(f"    筛选条件: {config['filter']}")
    
    return configs

if __name__ == "__main__":
    # 运行示例
    main()
    
    # 创建基准配置
    benchmark_configs = create_benchmark_dataset()
    
    print("\n💡 使用提示:")
    print("1. 根据实际需求调整筛选条件")
    print("2. 可以组合使用不同的预处理方法")
    print("3. 对于迁移学习，建议使用不同域的数据进行实验")
    print("4. 特征工程可以显著提升模型性能")
    print("5. 数据增强有助于提高模型泛化能力")

