#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
详细检查CRWU样本文件内容
"""

import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def check_sample_files():
    """检查各类别的样本文件"""
    
    crwu_path = Path("/Users/gsyoung/MBP_documents/code/Matlab/研究生数学建模/CRWU")
    
    # 选择代表性样本文件
    sample_files = {
        '12kHz_DE_Ball': crwu_path / '12k Drive End Bearing Fault Data/Ball/0007/B007_0.mat',
        '12kHz_FE_IR': crwu_path / '12k Fan End Bearing Fault Data/Inner Race/0007/IR007_0.mat',
        '48kHz_DE_OR': crwu_path / '48k Drive End Bearing Fault Data/Outer Race/Centered/0007/OR007@6_0.mat',
        'Normal': crwu_path / 'Normal Baseline/normal_0.mat'
    }
    
    print("🔍 CRWU详细样本分析")
    print("=" * 60)
    
    for category, file_path in sample_files.items():
        print(f"\n📋 {category}")
        print(f"文件: {file_path.name}")
        print("-" * 40)
        
        if file_path.exists():
            try:
                mat_data = sio.loadmat(str(file_path))
                keys = [k for k in mat_data.keys() if not k.startswith('__')]
                
                print(f"变量数量: {len(keys)}")
                
                for key in keys:
                    var = mat_data[key]
                    if isinstance(var, np.ndarray):
                        print(f"  {key}:")
                        print(f"    形状: {var.shape}")
                        print(f"    数据类型: {var.dtype}")
                        print(f"    数据范围: [{np.min(var):.6f}, {np.max(var):.6f}]")
                        
                        # 如果是时间序列数据，显示采样信息
                        if 'time' in key.lower() and len(var.shape) >= 1 and var.shape[0] > 1000:
                            print(f"    采样点数: {var.shape[0]}")
                            if '12k' in category:
                                duration = var.shape[0] / 12000
                                print(f"    时长: {duration:.2f} 秒 (假设12kHz)")
                            elif '48k' in category:
                                duration = var.shape[0] / 48000
                                print(f"    时长: {duration:.2f} 秒 (假设48kHz)")
                        
                        # 如果是RPM数据
                        if 'rpm' in key.lower():
                            print(f"    转速: {var.flatten()[0]} RPM")
                
            except Exception as e:
                print(f"❌ 错误: {e}")
        else:
            print("❌ 文件不存在")

def compare_data_formats():
    """对比CRWU与源域数据集的数据格式"""
    
    print(f"\n📊 数据格式对比")
    print("=" * 60)
    
    # CRWU样本
    crwu_sample = Path("/Users/gsyoung/MBP_documents/code/Matlab/研究生数学建模/CRWU/12k Drive End Bearing Fault Data/Ball/0007/B007_0.mat")
    
    # 源域样本
    source_sample = Path("/Users/gsyoung/MBP_documents/code/Matlab/研究生数学建模/数据集/源域数据集/12kHz_DE_data/B/0007/B007_0.mat")
    
    print("\n🔍 CRWU数据格式:")
    if crwu_sample.exists():
        try:
            crwu_data = sio.loadmat(str(crwu_sample))
            crwu_keys = [k for k in crwu_data.keys() if not k.startswith('__')]
            print(f"变量: {crwu_keys}")
            
            for key in crwu_keys:
                var = crwu_data[key]
                if isinstance(var, np.ndarray):
                    print(f"  {key}: {var.shape} ({var.dtype})")
                    
        except Exception as e:
            print(f"❌ 读取CRWU样本错误: {e}")
    
    print("\n🔍 源域数据格式:")
    if source_sample.exists():
        try:
            source_data = sio.loadmat(str(source_sample))
            source_keys = [k for k in source_data.keys() if not k.startswith('__')]
            print(f"变量: {source_keys}")
            
            for key in source_keys:
                var = source_data[key]
                if isinstance(var, np.ndarray):
                    print(f"  {key}: {var.shape} ({var.dtype})")
                    
        except Exception as e:
            print(f"❌ 读取源域样本错误: {e}")
    else:
        print("❌ 源域样本文件不存在")

def count_detailed_categories():
    """详细统计各类别数量"""
    
    print(f"\n📈 详细类别统计")
    print("=" * 60)
    
    crwu_path = Path("/Users/gsyoung/MBP_documents/code/Matlab/研究生数学建模/CRWU")
    
    # 统计各个故障类型和尺寸
    fault_stats = {
        '12kHz_DE': {'Ball': {}, 'Inner_Race': {}, 'Outer_Race': {}},
        '12kHz_FE': {'Ball': {}, 'Inner_Race': {}, 'Outer_Race': {}},
        '48kHz_DE': {'Ball': {}, 'Inner_Race': {}, 'Outer_Race': {}},
        'Normal': {}
    }
    
    # 12kHz Drive End
    de_12k_path = crwu_path / '12k Drive End Bearing Fault Data'
    if de_12k_path.exists():
        # Ball
        ball_path = de_12k_path / 'Ball'
        for size_dir in ball_path.iterdir():
            if size_dir.is_dir():
                size = size_dir.name
                files = list(size_dir.glob('*.mat'))
                fault_stats['12kHz_DE']['Ball'][size] = len(files)
        
        # Inner Race
        ir_path = de_12k_path / 'Inner Race'
        for size_dir in ir_path.iterdir():
            if size_dir.is_dir():
                size = size_dir.name
                files = list(size_dir.glob('*.mat'))
                fault_stats['12kHz_DE']['Inner_Race'][size] = len(files)
        
        # Outer Race
        or_path = de_12k_path / 'Outer Race'
        for position_dir in or_path.iterdir():
            if position_dir.is_dir():
                position = position_dir.name
                if position not in fault_stats['12kHz_DE']['Outer_Race']:
                    fault_stats['12kHz_DE']['Outer_Race'][position] = {}
                for size_dir in position_dir.iterdir():
                    if size_dir.is_dir():
                        size = size_dir.name
                        files = list(size_dir.glob('*.mat'))
                        fault_stats['12kHz_DE']['Outer_Race'][position][size] = len(files)
    
    # 类似地处理其他类别...
    
    # 打印统计结果
    print("📊 12kHz Drive End (DE):")
    for fault_type, sizes in fault_stats['12kHz_DE'].items():
        if sizes:
            print(f"  {fault_type}:")
            if fault_type == 'Outer_Race':
                for position, pos_sizes in sizes.items():
                    print(f"    {position}:")
                    for size, count in pos_sizes.items():
                        print(f"      {size}: {count} 个文件")
            else:
                for size, count in sizes.items():
                    print(f"    {size}: {count} 个文件")
    
    # Normal数据
    normal_path = crwu_path / 'Normal Baseline'
    if normal_path.exists():
        normal_files = list(normal_path.glob('*.mat'))
        print(f"\n📊 Normal Baseline: {len(normal_files)} 个文件")

def main():
    """主函数"""
    check_sample_files()
    compare_data_formats() 
    count_detailed_categories()

if __name__ == "__main__":
    main()
