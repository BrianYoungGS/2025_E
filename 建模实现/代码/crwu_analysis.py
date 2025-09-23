#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CRWU数据集分析脚本
分析CRWU数据集与源域数据集的异同，评估是否可作为扩展数据集
"""

import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import json

def analyze_crwu_dataset():
    """分析CRWU数据集"""
    
    crwu_path = Path("/Users/gsyoung/MBP_documents/code/Matlab/研究生数学建模/CRWU")
    
    print("🔍 CRWU数据集分析")
    print("=" * 60)
    
    # 数据集统计
    categories = {
        '12k Drive End Bearing Fault Data': '12kHz_DE',
        '12k Fan End Bearing Fault Data': '12kHz_FE',
        '48k Drive End Bearing Fault Data': '48kHz_DE',
        'Normal Baseline': 'Normal'
    }
    
    dataset_stats = {}
    total_files = 0
    
    for category, code in categories.items():
        category_path = crwu_path / category
        if category_path.exists():
            mat_files = list(category_path.rglob("*.mat"))
            file_count = len(mat_files)
            dataset_stats[code] = {
                'count': file_count,
                'files': mat_files
            }
            total_files += file_count
            print(f"📁 {category}: {file_count} 个文件")
    
    print(f"\n📊 总计: {total_files} 个.mat文件")
    
    return dataset_stats, total_files

def analyze_sample_files(dataset_stats):
    """分析样本文件格式"""
    
    print(f"\n🔬 样本文件分析")
    print("-" * 40)
    
    sample_info = {}
    
    for category, data in dataset_stats.items():
        if data['files']:
            # 取第一个文件作为样本
            sample_file = data['files'][0]
            print(f"\n📋 {category} 样本: {sample_file.name}")
            
            try:
                mat_data = sio.loadmat(str(sample_file))
                keys = [k for k in mat_data.keys() if not k.startswith('__')]
                
                file_info = {
                    'path': str(sample_file),
                    'variables': {}
                }
                
                for key in keys:
                    var = mat_data[key]
                    if isinstance(var, np.ndarray):
                        file_info['variables'][key] = {
                            'shape': var.shape,
                            'dtype': str(var.dtype),
                            'size': var.size
                        }
                        print(f"  {key}: {var.shape} ({var.dtype})")
                
                sample_info[category] = file_info
                
            except Exception as e:
                print(f"  ❌ 读取错误: {e}")
                sample_info[category] = {'error': str(e)}
    
    return sample_info

def compare_with_source_dataset():
    """与源域数据集对比"""
    
    print(f"\n📈 与源域数据集对比")
    print("-" * 40)
    
    # 源域数据集统计（基于之前的处理结果）
    source_stats = {
        '12kHz_DE': 60,   # 从处理报告中得到
        '12kHz_FE': 45,
        '48kHz_DE': 52,
        '48kHz_Normal': 4,
        'Total': 161
    }
    
    print("源域数据集:")
    for category, count in source_stats.items():
        print(f"  {category}: {count} 个文件")
    
    return source_stats

def evaluate_compatibility(crwu_stats, source_stats, sample_info):
    """评估兼容性"""
    
    print(f"\n⚖️ 兼容性评估")
    print("-" * 40)
    
    compatibility = {
        'file_format': True,
        'data_structure': True,
        'categories_match': True,
        'can_extend': True,
        'issues': []
    }
    
    # 1. 检查数据类别
    crwu_categories = set(crwu_stats.keys())
    source_categories = set([k for k in source_stats.keys() if k != 'Total'])
    
    print("📋 数据类别对比:")
    for cat in source_categories:
        if cat in crwu_categories:
            print(f"  ✅ {cat}: 两个数据集都有")
        else:
            print(f"  ❌ {cat}: CRWU中缺失")
            compatibility['issues'].append(f"CRWU缺少{cat}类别")
    
    # 2. 检查文件格式
    print(f"\n📄 文件格式检查:")
    all_mat = True
    for category, info in sample_info.items():
        if 'error' in info:
            print(f"  ❌ {category}: 文件读取错误")
            all_mat = False
            compatibility['file_format'] = False
            compatibility['issues'].append(f"{category}文件格式问题")
        else:
            print(f"  ✅ {category}: .mat格式正常")
    
    # 3. 检查数据结构
    print(f"\n🔧 数据结构检查:")
    expected_vars = ['DE_time', 'FE_time', 'BA_time', 'RPM']
    
    for category, info in sample_info.items():
        if 'variables' in info:
            vars_found = list(info['variables'].keys())
            has_de = any('DE' in var for var in vars_found)
            has_time = any('time' in var for var in vars_found)
            
            if has_de and has_time:
                print(f"  ✅ {category}: 包含时域数据")
            else:
                print(f"  ⚠️ {category}: 数据结构可能不同")
                print(f"      变量: {vars_found}")
                compatibility['issues'].append(f"{category}数据结构异常")
    
    # 4. 数量对比
    print(f"\n📊 数量对比:")
    total_crwu = sum(stats['count'] for stats in crwu_stats.values())
    total_source = source_stats['Total']
    
    print(f"  CRWU数据集: {total_crwu} 个文件")
    print(f"  源域数据集: {total_source} 个文件")
    print(f"  扩展后总数: {total_crwu + total_source} 个文件")
    
    # 5. 最终评估
    if len(compatibility['issues']) == 0:
        compatibility['can_extend'] = True
        print(f"\n✅ 兼容性评估: 可以作为扩展数据集使用")
    else:
        compatibility['can_extend'] = False
        print(f"\n⚠️ 兼容性评估: 需要处理以下问题才能使用:")
        for issue in compatibility['issues']:
            print(f"    - {issue}")
    
    return compatibility

def generate_crwu_report(crwu_stats, source_stats, sample_info, compatibility):
    """生成CRWU分析报告"""
    
    report = {
        'analysis_time': pd.Timestamp.now().isoformat(),
        'crwu_dataset': crwu_stats,
        'source_dataset': source_stats,
        'sample_analysis': sample_info,
        'compatibility': compatibility,
        'recommendations': []
    }
    
    # 生成建议
    if compatibility['can_extend']:
        report['recommendations'] = [
            "✅ 可以直接使用CRWU数据集作为扩展",
            "✅ 使用现有的数据处理管道处理CRWU数据",
            "✅ 合并后可获得更大的训练数据集",
            "✅ 保持相同的特征提取和处理流程"
        ]
    else:
        report['recommendations'] = [
            "⚠️ 需要先解决兼容性问题",
            "⚠️ 可能需要调整数据处理流程",
            "⚠️ 建议先处理少量样本验证",
            "⚠️ 确保数据质量和一致性"
        ]
    
    # 保存报告
    base_path = Path("/Users/gsyoung/MBP_documents/code/Matlab/研究生数学建模/建模实现/处理后数据/reports")
    
    # JSON报告
    with open(base_path / "crwu_analysis_report.json", 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    
    # 文本报告
    with open(base_path / "crwu_analysis_report.txt", 'w', encoding='utf-8') as f:
        f.write("CRWU数据集分析报告\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("📊 数据集对比:\n")
        f.write(f"CRWU数据集总文件数: {sum(stats['count'] for stats in crwu_stats.values())}\n")
        f.write(f"源域数据集总文件数: {source_stats['Total']}\n")
        f.write(f"合并后总文件数: {sum(stats['count'] for stats in crwu_stats.values()) + source_stats['Total']}\n\n")
        
        f.write("📋 类别分布对比:\n")
        for category in ['12kHz_DE', '12kHz_FE', '48kHz_DE', 'Normal']:
            crwu_count = crwu_stats.get(category, {}).get('count', 0)
            source_count = source_stats.get(category, 0)
            f.write(f"  {category}: CRWU={crwu_count}, 源域={source_count}\n")
        
        f.write("\n🔧 兼容性评估:\n")
        if compatibility['can_extend']:
            f.write("✅ 可以作为扩展数据集使用\n")
        else:
            f.write("⚠️ 需要处理兼容性问题\n")
            for issue in compatibility['issues']:
                f.write(f"  - {issue}\n")
        
        f.write("\n📝 建议:\n")
        for rec in report['recommendations']:
            f.write(f"  {rec}\n")
    
    print(f"\n📄 分析报告已保存到:")
    print(f"  - crwu_analysis_report.json")
    print(f"  - crwu_analysis_report.txt")

def main():
    """主函数"""
    print("CRWU数据集分析工具")
    print("=" * 60)
    
    # 分析CRWU数据集
    crwu_stats, total_files = analyze_crwu_dataset()
    
    # 分析样本文件
    sample_info = analyze_sample_files(crwu_stats)
    
    # 与源域数据集对比
    source_stats = compare_with_source_dataset()
    
    # 兼容性评估
    compatibility = evaluate_compatibility(crwu_stats, source_stats, sample_info)
    
    # 生成报告
    generate_crwu_report(crwu_stats, source_stats, sample_info, compatibility)

if __name__ == "__main__":
    main()
