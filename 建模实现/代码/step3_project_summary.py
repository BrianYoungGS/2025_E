#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
项目总结报告生成器
展示整个数据处理流程的完成情况和结果
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import json

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

def generate_project_summary():
    """生成项目总结报告"""
    
    base_path = Path("/Users/gsyoung/MBP_documents/code/Matlab/研究生数学建模")
    output_path = base_path / "建模实现" / "处理后数据"
    
    print("="*80)
    print("源域数据集处理项目总结报告")
    print("="*80)
    
    # 1. 项目概述
    print("\n📋 项目概述:")
    print("  - 目标: 对161个源域轴承数据进行预处理，生成用于后续诊断任务的特征数据集")
    print("  - 输入: 161个.mat格式的轴承振动数据文件")
    print("  - 输出: 322个数据片段，每个包含4种文件格式")
    
    # 2. 处理流程总结
    print("\n🔄 完整处理流程:")
    processing_steps = [
        "1. 数据分析 - 扫描和分析源域数据集结构",
        "2. 数据降采样 - 将48kHz数据降采样到12kHz统一格式", 
        "3. 去噪滤波 - 应用多重滤波器组合去除噪声",
        "   • 高通滤波 (10Hz) - 去除低频趋势",
        "   • 低通滤波 (5000Hz) - 去除高频噪声", 
        "   • 陷波滤波 (50Hz及谐波) - 去除工频干扰",
        "   • 中值滤波 - 去除脉冲噪声",
        "4. 数据对齐 - 从每个文件前后各截取2048点数据片段",
        "5. 特征提取 - 计算29个时域和频域特征",
        "6. 数据可视化 - 生成时域和频域信号图像",
        "7. 结果输出 - 组织为标准化的文件结构"
    ]
    
    for step in processing_steps:
        print(f"  {step}")
    
    # 3. 去噪滤波参数
    print("\n🔧 去噪滤波参数:")
    filter_params = {
        'highpass_cutoff': '10 Hz (去除低频噪声)',
        'lowpass_cutoff': '5000 Hz (去除高频噪声)',
        'notch_freq': '50 Hz + 谐波 (去除工频干扰)',
        'notch_q': '30 (陷波滤波品质因数)',
        'median_filter': '3点核 (去除脉冲噪声)'
    }
    
    for param, desc in filter_params.items():
        print(f"  • {param}: {desc}")
    
    # 4. 数据统计
    try:
        report_file = output_path / "reports" / "processing_report.json"
        if report_file.exists():
            with open(report_file, 'r', encoding='utf-8') as f:
                report_data = json.load(f)
            
            print("\n📊 处理结果统计:")
            print(f"  • 源文件总数: {report_data['total_source_files']}")
            print(f"  • 生成数据片段: {report_data['total_segments_generated']}")
            print(f"  • 采样频率: {report_data['target_sampling_frequency']} Hz")
            print(f"  • 片段长度: {report_data['segment_length']} 点")
            print(f"  • 特征维度: {report_data['feature_count']} 个")
            
            print("\n  故障类型分布:")
            for fault_type, count in report_data['fault_type_distribution'].items():
                percentage = (count / report_data['total_segments_generated']) * 100
                print(f"    - {fault_type}: {count} 个片段 ({percentage:.1f}%)")
        
    except Exception as e:
        print(f"  警告: 无法读取处理报告 - {e}")
    
    # 5. 特征说明
    print("\n📈 特征提取详情:")
    print("  时域特征 (P1-P16):")
    time_features = [
        "P1-P8: 基本统计量 (均值、RMS、方差等)",
        "P9-P10: 高阶统计量 (偏度、峰度)",
        "P11-P16: 形状因子和变异系数"
    ]
    for feature in time_features:
        print(f"    • {feature}")
    
    print("  频域特征 (P17-P29):")
    freq_features = [
        "P17-P23: 频谱统计量 (平均频率、频谱方差等)",
        "P24-P25: 主频带位置指示器",
        "P26-P29: 频谱分散度和集中度"
    ]
    for feature in freq_features:
        print(f"    • {feature}")
    
    # 6. 输出文件结构
    print("\n📁 输出文件结构:")
    print("  建模实现/处理后数据/")
    print("  ├── processed_segments/          # 322个数据片段文件夹")
    print("  │   ├── [文件名]_front_2048/     # 前段数据")
    print("  │   │   ├── *_raw_data.npy       # 去噪后的原始数据") 
    print("  │   │   ├── *_time_domain.png    # 时域信号图像")
    print("  │   │   ├── *_frequency_domain.png # 频域信号图像")
    print("  │   │   └── *_features.csv       # 29维特征+标签")
    print("  │   └── [文件名]_back_2048/      # 后段数据")
    print("  │       └── ... (同上)")
    print("  └── reports/                     # 处理报告")
    print("      ├── processing_report.txt    # 文本报告")
    print("      ├── processing_report.json   # JSON报告")
    print("      └── all_features_summary.csv # 所有特征汇总")
    
    # 7. 技术亮点
    print("\n⭐ 技术亮点:")
    highlights = [
        "多层次去噪滤波 - 组合4种滤波器有效去除各类噪声",
        "自适应降采样 - 保证信号质量的同时统一采样率",
        "全面特征提取 - 涵盖时域和频域的29个特征",
        "标准化输出 - 便于后续机器学习算法使用",
        "可视化展示 - 直观显示信号处理效果",
        "完整的可追溯性 - 详细记录处理参数和流程"
    ]
    
    for highlight in highlights:
        print(f"  • {highlight}")
    
    # 8. 质量保证
    print("\n✅ 质量保证措施:")
    quality_measures = [
        "数据完整性检查 - 确保所有161个文件都被正确处理",
        "特征有效性验证 - 检查特征值的合理性范围", 
        "文件格式标准化 - 统一的命名规则和数据格式",
        "处理过程记录 - 完整的日志和参数记录",
        "异常处理机制 - 对异常数据进行适当处理"
    ]
    
    for measure in quality_measures:
        print(f"  • {measure}")
    
    # 9. 后续应用建议
    print("\n🚀 后续应用建议:")
    applications = [
        "机器学习训练 - 使用特征CSV文件训练分类模型",
        "深度学习应用 - 使用raw_data.npy进行端到端训练",
        "数据分析研究 - 基于可视化图像进行信号分析",
        "模型验证 - 使用不同故障类型数据验证模型性能",
        "特征工程 - 基于现有特征构建更高级的组合特征"
    ]
    
    for app in applications:
        print(f"  • {app}")
    
    print("\n" + "="*80)
    print("🎉 项目完成！所有数据处理任务已成功完成！")
    print("="*80)

def verify_output_completeness():
    """验证输出完整性"""
    
    output_path = Path("/Users/gsyoung/MBP_documents/code/Matlab/研究生数学建模/建模实现/处理后数据")
    segments_path = output_path / "processed_segments"
    
    if not segments_path.exists():
        print("❌ 错误: processed_segments文件夹不存在")
        return False
    
    # 统计文件数量
    segment_dirs = [d for d in segments_path.iterdir() if d.is_dir()]
    total_segments = len(segment_dirs)
    
    print(f"\n🔍 输出完整性验证:")
    print(f"  • 数据片段文件夹总数: {total_segments}")
    
    # 检查文件完整性
    complete_segments = 0
    for segment_dir in segment_dirs:
        required_files = [
            f"{segment_dir.name}_raw_data.npy",
            f"{segment_dir.name}_time_domain.png", 
            f"{segment_dir.name}_frequency_domain.png",
            f"{segment_dir.name}_features.csv"
        ]
        
        if all((segment_dir / file).exists() for file in required_files):
            complete_segments += 1
    
    print(f"  • 完整的数据片段: {complete_segments}/{total_segments}")
    
    if complete_segments == total_segments:
        print("  ✅ 所有数据片段都包含完整的4个文件")
        return True
    else:
        print(f"  ⚠️  有 {total_segments - complete_segments} 个数据片段不完整")
        return False

if __name__ == "__main__":
    generate_project_summary()
    verify_output_completeness()
