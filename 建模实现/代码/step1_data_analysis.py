#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
源域数据集分析脚本
分析源域数据集的结构和内容，为后续处理做准备
"""

import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

def analyze_mat_file(file_path):
    """
    分析单个.mat文件的内容
    
    Args:
        file_path: .mat文件路径
        
    Returns:
        dict: 包含文件信息的字典
    """
    try:
        data = sio.loadmat(file_path)
        
        # 过滤掉MATLAB的元数据
        keys = [k for k in data.keys() if not k.startswith('__')]
        
        file_info = {
            'path': file_path,
            'variables': {},
            'sampling_freq': None,
            'rpm': None
        }
        
        for key in keys:
            var = data[key]
            if isinstance(var, np.ndarray):
                file_info['variables'][key] = {
                    'shape': var.shape,
                    'dtype': str(var.dtype),
                    'size': var.size
                }
                
                # 提取采样频率和转速信息
                if 'RPM' in key.upper():
                    file_info['rpm'] = var[0] if var.size > 0 else None
                    
        return file_info
        
    except Exception as e:
        print(f"分析文件 {file_path} 时出错: {e}")
        return None

def scan_source_dataset(base_path):
    """
    扫描整个源域数据集
    
    Args:
        base_path: 源域数据集根目录
        
    Returns:
        dict: 数据集统计信息
    """
    source_path = Path(base_path) / "数据集" / "数据集" / "源域数据集"
    
    dataset_info = {
        '12kHz_DE_data': {'count': 0, 'files': []},
        '12kHz_FE_data': {'count': 0, 'files': []},
        '48kHz_DE_data': {'count': 0, 'files': []},
        '48kHz_Normal_data': {'count': 0, 'files': []},
        'total_files': 0
    }
    
    print("正在扫描源域数据集...")
    
    for category in dataset_info.keys():
        if category == 'total_files':
            continue
            
        category_path = source_path / category
        if category_path.exists():
            print(f"\n分析 {category} 类别:")
            
            for root, dirs, files in os.walk(category_path):
                for file in files:
                    if file.endswith('.mat'):
                        file_path = Path(root) / file
                        file_info = analyze_mat_file(str(file_path))
                        
                        if file_info:
                            dataset_info[category]['files'].append(file_info)
                            dataset_info[category]['count'] += 1
                            dataset_info['total_files'] += 1
                            
                            # 打印文件信息
                            rel_path = file_path.relative_to(source_path)
                            print(f"  {rel_path}: ", end="")
                            for var_name, var_info in file_info['variables'].items():
                                print(f"{var_name}{var_info['shape']}", end=" ")
                            if file_info['rpm'] is not None:
                                rpm_val = float(file_info['rpm']) if hasattr(file_info['rpm'], '__iter__') else file_info['rpm']
                                print(f"RPM={rpm_val:.0f}", end="")
                            print()
    
    return dataset_info

def analyze_data_characteristics(dataset_info):
    """
    分析数据特征
    
    Args:
        dataset_info: 数据集信息字典
    """
    print("\n" + "="*50)
    print("数据集统计分析")
    print("="*50)
    
    for category, info in dataset_info.items():
        if category == 'total_files':
            continue
            
        print(f"\n{category}:")
        print(f"  文件数量: {info['count']}")
        
        if info['files']:
            # 分析第一个文件的变量结构
            sample_file = info['files'][0]
            print(f"  变量结构 (以 {Path(sample_file['path']).name} 为例):")
            for var_name, var_info in sample_file['variables'].items():
                print(f"    {var_name}: {var_info['shape']} ({var_info['dtype']})")
            
            # 统计转速信息
            rpms = [float(f['rpm']) if f['rpm'] is not None else None for f in info['files']]
            rpms = [r for r in rpms if r is not None]
            if rpms:
                print(f"  转速范围: {min(rpms):.0f} - {max(rpms):.0f} RPM")
                print(f"  不同转速数量: {len(set([round(r) for r in rpms]))}")
    
    print(f"\n总文件数: {dataset_info['total_files']}")

def create_processing_plan(dataset_info):
    """
    创建数据处理计划
    
    Args:
        dataset_info: 数据集信息字典
    """
    print("\n" + "="*50)
    print("数据处理计划")
    print("="*50)
    
    print("\n1. 降采样需求分析:")
    print(f"   - 48kHz数据: {dataset_info['48kHz_DE_data']['count']} + {dataset_info['48kHz_Normal_data']['count']} = {dataset_info['48kHz_DE_data']['count'] + dataset_info['48kHz_Normal_data']['count']} 个文件")
    print(f"   - 12kHz数据: {dataset_info['12kHz_DE_data']['count']} + {dataset_info['12kHz_FE_data']['count']} = {dataset_info['12kHz_DE_data']['count'] + dataset_info['12kHz_FE_data']['count']} 个文件")
    print("   - 建议: 将48kHz数据降采样到12kHz以保持一致性")
    
    print("\n2. 数据对齐计划:")
    total_files = dataset_info['total_files']
    print(f"   - 总文件数: {total_files}")
    print(f"   - 预期输出: {total_files * 2} 个数据片段 (每个文件前后各取一段)")
    
    print("\n3. 特征提取计划:")
    print("   - 时域特征: 统计特征 (P1-P16)")
    print("   - 频域特征: 频谱特征 (P17-P29)")
    print("   - 输出格式: CSV文件包含29个特征 + 故障类型标签")

def main():
    """主函数"""
    # 数据集根路径
    base_path = "/Users/gsyoung/MBP_documents/code/Matlab/研究生数学建模"
    
    print("源域数据集分析工具")
    print("="*50)
    
    # 扫描数据集
    dataset_info = scan_source_dataset(base_path)
    
    # 分析数据特征
    analyze_data_characteristics(dataset_info)
    
    # 创建处理计划
    create_processing_plan(dataset_info)
    
    # 保存分析结果
    output_path = Path(base_path) / "建模实现" / "处理后数据" / "dataset_analysis.txt"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("源域数据集分析结果\n")
        f.write("="*30 + "\n\n")
        
        for category, info in dataset_info.items():
            if category == 'total_files':
                continue
            f.write(f"{category}: {info['count']} 个文件\n")
        
        f.write(f"\n总文件数: {dataset_info['total_files']}\n")
    
    print(f"\n分析结果已保存到: {output_path}")

if __name__ == "__main__":
    main()
