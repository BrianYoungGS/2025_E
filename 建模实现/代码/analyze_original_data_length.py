#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析原始数据长度分布，设计最佳分段策略
重新审视数据处理需求：从每个mat文件生成3个有效数据片段
"""

import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import json

def analyze_original_data_lengths():
    """分析源域数据集中所有文件的数据长度"""
    
    source_base = Path("/Users/gsyoung/MBP_documents/code/Matlab/研究生数学建模/数据集/数据集/源域数据集")
    
    print("🔍 原始数据长度分析")
    print("=" * 60)
    
    length_stats = {
        'files': [],
        'lengths': [],
        'categories': [],
        'sampling_rates': []
    }
    
    # 遍历所有数据集类别
    categories = {
        '12kHz_DE_data': 12000,
        '12kHz_FE_data': 12000,
        '48kHz_DE_data': 48000,
        '48kHz_Normal_data': 48000
    }
    
    for category, fs in categories.items():
        category_path = source_base / category
        if not category_path.exists():
            print(f"⚠️ 路径不存在: {category_path}")
            continue
            
        print(f"\n📁 分析类别: {category}")
        
        # 查找所有.mat文件
        mat_files = list(category_path.rglob("*.mat"))
        print(f"找到 {len(mat_files)} 个.mat文件")
        
        for i, mat_file in enumerate(mat_files[:5]):  # 先分析前5个文件
            print(f"\n  📄 文件 {i+1}: {mat_file.name}")
            
            try:
                mat_data = sio.loadmat(str(mat_file))
                
                # 查找时域数据变量
                time_vars = [k for k in mat_data.keys() 
                           if not k.startswith('__') and 'time' in k.lower()]
                
                for var_name in time_vars:
                    data = mat_data[var_name]
                    if isinstance(data, np.ndarray) and len(data.shape) >= 1:
                        length = data.shape[0]
                        duration = length / fs
                        
                        print(f"    {var_name}: {length} 点 ({duration:.2f}秒)")
                        
                        length_stats['files'].append(str(mat_file))
                        length_stats['lengths'].append(length)
                        length_stats['categories'].append(category)
                        length_stats['sampling_rates'].append(fs)
                        
            except Exception as e:
                print(f"    ❌ 读取错误: {e}")
    
    return length_stats

def design_segmentation_strategy(length_stats):
    """设计数据分段策略"""
    
    print(f"\n📊 数据分段策略设计")
    print("-" * 40)
    
    # 统计数据长度分布
    lengths = np.array(length_stats['lengths'])
    categories = length_stats['categories']
    sampling_rates = length_stats['sampling_rates']
    
    print(f"数据长度统计:")
    print(f"  最小长度: {np.min(lengths):,} 点")
    print(f"  最大长度: {np.max(lengths):,} 点")
    print(f"  平均长度: {np.mean(lengths):,.0f} 点")
    print(f"  中位长度: {np.median(lengths):,.0f} 点")
    
    # 按类别统计
    print(f"\n按类别统计:")
    df = pd.DataFrame({
        'length': lengths,
        'category': categories,
        'sampling_rate': sampling_rates
    })
    
    category_stats = df.groupby(['category', 'sampling_rate']).agg({
        'length': ['count', 'min', 'max', 'mean', 'median']
    }).round(0)
    
    print(category_stats)
    
    # 设计分段策略
    print(f"\n🎯 分段策略设计:")
    
    strategies = {}
    
    for (category, fs), group in df.groupby(['category', 'sampling_rate']):
        min_length = group['length'].min()
        avg_length = group['length'].mean()
        
        # 需求：每段>=6万点，降采样后>=2万点
        min_segment_points = 60000  # 原始数据最小点数
        min_downsampled_points = 20000  # 降采样后最小点数
        
        # 如果是48kHz数据，需要降采样到12kHz
        if fs == 48000:
            target_fs = 12000
            downsample_ratio = fs / target_fs  # 4:1
            min_original_points = min_downsampled_points * downsample_ratio  # 80000点
        else:
            target_fs = fs
            downsample_ratio = 1
            min_original_points = min_downsampled_points  # 20000点
        
        # 确保满足最小要求
        actual_min_points = max(min_segment_points, min_original_points)
        
        # 计算可以分成几段
        num_segments = int(min_length // actual_min_points)
        segment_length = int(min_length // num_segments) if num_segments > 0 else min_length
        
        # 但我们要求是3段，所以重新计算
        target_segments = 3
        required_total_length = target_segments * actual_min_points
        
        if min_length >= required_total_length:
            # 可以分成3段
            segment_length = int(min_length // target_segments)
            feasible = True
        else:
            # 数据不够分成3段，调整策略
            segment_length = int(min_length // 2)  # 分成2段
            target_segments = 2
            feasible = False
        
        strategies[f"{category}_{fs}Hz"] = {
            'category': category,
            'original_fs': int(fs),
            'target_fs': int(target_fs),
            'downsample_ratio': float(downsample_ratio),
            'min_length': int(min_length),
            'avg_length': int(avg_length),
            'target_segments': int(target_segments),
            'segment_length': int(segment_length),
            'min_required_points': int(actual_min_points),
            'feasible_3_segments': bool(feasible),
            'downsampled_segment_length': int(segment_length / downsample_ratio)
        }
        
        print(f"\n  📊 {category} ({fs}Hz):")
        print(f"    原始长度范围: {int(min_length):,} - {int(group['length'].max()):,} 点")
        print(f"    降采样比例: {downsample_ratio:.1f}:1 ({fs}Hz → {target_fs}Hz)")
        print(f"    建议分段数: {target_segments} 段")
        print(f"    每段长度: {segment_length:,} 点")
        print(f"    降采样后每段: {int(segment_length / downsample_ratio):,} 点")
        print(f"    3段方案可行: {'✅' if feasible else '❌'}")
    
    return strategies

def visualize_length_distribution(length_stats):
    """可视化数据长度分布"""
    
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('原始数据长度分布分析', fontsize=16, fontweight='bold')
    
    df = pd.DataFrame({
        'length': length_stats['lengths'],
        'category': length_stats['categories'],
        'sampling_rate': length_stats['sampling_rates']
    })
    
    # 1. 总体长度分布
    axes[0, 0].hist(df['length'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].set_title('总体数据长度分布')
    axes[0, 0].set_xlabel('数据长度 (点)')
    axes[0, 0].set_ylabel('文件数量')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 按类别分布
    categories = df['category'].unique()
    colors = ['red', 'blue', 'green', 'orange']
    for i, cat in enumerate(categories):
        cat_data = df[df['category'] == cat]['length']
        axes[0, 1].hist(cat_data, bins=10, alpha=0.6, 
                       color=colors[i % len(colors)], label=cat)
    axes[0, 1].set_title('按类别的长度分布')
    axes[0, 1].set_xlabel('数据长度 (点)')
    axes[0, 1].set_ylabel('文件数量')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 箱线图
    box_data = [df[df['category'] == cat]['length'] for cat in categories]
    axes[1, 0].boxplot(box_data, tick_labels=[cat.replace('_data', '') for cat in categories])
    axes[1, 0].set_title('各类别长度分布箱线图')
    axes[1, 0].set_ylabel('数据长度 (点)')
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. 统计摘要表
    axes[1, 1].axis('off')
    summary_stats = df.groupby('category')['length'].describe()
    table_data = []
    for cat in summary_stats.index:
        stats = summary_stats.loc[cat]
        table_data.append([
            cat.replace('_data', ''),
            f"{stats['count']:.0f}",
            f"{stats['min']:,.0f}",
            f"{stats['max']:,.0f}",
            f"{stats['mean']:,.0f}",
            f"{stats['std']:,.0f}"
        ])
    
    table = axes[1, 1].table(cellText=table_data,
                            colLabels=['类别', '文件数', '最小值', '最大值', '平均值', '标准差'],
                            cellLoc='center',
                            loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    axes[1, 1].set_title('统计摘要表')
    
    plt.tight_layout()
    
    # 保存图像
    output_dir = Path("/Users/gsyoung/MBP_documents/code/Matlab/研究生数学建模/建模实现/处理后数据/reports")
    plt.savefig(output_dir / "original_data_length_analysis.png", dpi=300, bbox_inches='tight')
    plt.show()

def save_analysis_results(length_stats, strategies):
    """保存分析结果"""
    
    output_dir = Path("/Users/gsyoung/MBP_documents/code/Matlab/研究生数学建模/建模实现/处理后数据/reports")
    
    # 保存详细数据
    df = pd.DataFrame(length_stats)
    df.to_csv(output_dir / "original_data_length_stats.csv", index=False, encoding='utf-8-sig')
    
    # 保存策略
    with open(output_dir / "segmentation_strategies.json", 'w', encoding='utf-8') as f:
        json.dump(strategies, f, ensure_ascii=False, indent=2)
    
    # 保存策略摘要
    strategy_df = pd.DataFrame(strategies).T
    strategy_df.to_csv(output_dir / "segmentation_strategies.csv", encoding='utf-8-sig')
    
    print(f"\n📄 分析结果已保存:")
    print(f"  - original_data_length_stats.csv: 详细长度数据")
    print(f"  - segmentation_strategies.json: 分段策略")
    print(f"  - segmentation_strategies.csv: 策略摘要")
    print(f"  - original_data_length_analysis.png: 可视化图表")

def main():
    """主函数"""
    print("重新设计数据处理方案")
    print("=" * 60)
    print("目标: 从每个mat文件生成3个数据片段")
    print("要求: 每段>=6万点，降采样后>=2万点")
    print("=" * 60)
    
    # 分析原始数据长度
    length_stats = analyze_original_data_lengths()
    
    if not length_stats['lengths']:
        print("❌ 未找到有效数据，请检查数据路径")
        return
    
    # 设计分段策略
    strategies = design_segmentation_strategy(length_stats)
    
    # 可视化分析
    visualize_length_distribution(length_stats)
    
    # 保存结果
    save_analysis_results(length_stats, strategies)
    
    print(f"\n✅ 分析完成！")
    print(f"下一步: 基于分析结果实现新的数据处理流程")

if __name__ == "__main__":
    main()
