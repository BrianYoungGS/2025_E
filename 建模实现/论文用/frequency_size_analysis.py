#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bearing Fault Data Analysis - Frequency and Size Based Bar Chart
"""

import os
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

def classify_files_by_frequency_and_size(data_path):
    """按采样频率和故障尺寸分类统计数据文件"""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data path not found: {data_path}")
    
    # 获取所有文件夹名称
    all_folders = [f for f in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, f))]
    
    # 过滤掉带有_denoised后缀的文件夹
    original_folders = [f for f in all_folders if not f.endswith('_denoised')]
    
    # 初始化统计字典
    freq_data = {
        '12k': {'B': defaultdict(int), 'OR': defaultdict(int), 'IR': defaultdict(int), 'N': defaultdict(int)},
        '48k': {'B': defaultdict(int), 'OR': defaultdict(int), 'IR': defaultdict(int), 'N': defaultdict(int)}
    }
    
    for folder in original_folders:
        folder_upper = folder.upper()
        
        # 确定采样频率
        if folder.startswith('12k_'):
            freq = '12k'
        elif folder.startswith('48k_'):
            freq = '48k'
        else:
            continue
        
        # 确定故障类型和尺寸
        if '_B' in folder_upper:
            fault_type = 'B'
            # 提取尺寸信息
            if 'B007' in folder_upper:
                size = '007'
            elif 'B014' in folder_upper:
                size = '014'
            elif 'B021' in folder_upper:
                size = '021'
            elif 'B028' in folder_upper:
                size = '028'
            else:
                size = 'Other'
            freq_data[freq][fault_type][size] += 1
            
        elif '_OR' in folder_upper:
            fault_type = 'OR'
            if 'OR007' in folder_upper:
                size = '007'
            elif 'OR014' in folder_upper:
                size = '014'
            elif 'OR021' in folder_upper:
                size = '021'
            else:
                size = 'Other'
            freq_data[freq][fault_type][size] += 1
            
        elif '_IR' in folder_upper:
            fault_type = 'IR'
            if 'IR007' in folder_upper:
                size = '007'
            elif 'IR014' in folder_upper:
                size = '014'
            elif 'IR021' in folder_upper:
                size = '021'
            elif 'IR028' in folder_upper:
                size = '028'
            else:
                size = 'Other'
            freq_data[freq][fault_type][size] += 1
            
        elif '_N' in folder_upper or 'NORMAL' in folder_upper:
            fault_type = 'N'
            size = 'Normal'
            freq_data[freq][fault_type][size] += 1
    
    return freq_data

def create_frequency_size_bar_chart(freq_data, output_path):
    """创建按频率和尺寸分类的柱状图"""
    
    # 设置图形参数
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # 定义颜色和标签
    fault_types = ['B', 'OR', 'IR', 'N']
    fault_labels = ['Ball Bearing', 'Outer Race', 'Inner Race', 'Normal']
    
    # 定义尺寸顺序
    size_order = ['007', '014', '021', '028', 'Normal', 'Other']
    
    def plot_frequency_data(ax, freq, title):
        """绘制单个频率的数据"""
        data = freq_data[freq]
        
        # 计算每个故障类型的位置
        x_positions = np.arange(len(fault_types))
        bar_width = 0.15
        
        # 收集所有出现的尺寸
        all_sizes = set()
        for fault_type in fault_types:
            all_sizes.update(data[fault_type].keys())
        
        # 按预定义顺序排序尺寸
        present_sizes = [size for size in size_order if size in all_sizes]
        
        # 为每个尺寸分配颜色
        size_colors = plt.cm.Set3(np.linspace(0, 1, len(present_sizes)))
        
        # 绘制每个尺寸的柱子
        for i, size in enumerate(present_sizes):
            values = []
            for fault_type in fault_types:
                values.append(data[fault_type].get(size, 0))
            
            # 计算柱子位置
            positions = x_positions + (i - len(present_sizes)/2 + 0.5) * bar_width
            
            bars = ax.bar(positions, values, bar_width, 
                         color=size_colors[i], alpha=0.8, 
                         label=f'Size {size}' if size != 'Normal' else 'Normal')
            
            # 添加数值标签
            for j, (pos, val) in enumerate(zip(positions, values)):
                if val > 0:
                    ax.text(pos, val + 0.5, str(val), ha='center', va='bottom', fontsize=9)
        
        # 设置图表属性
        ax.set_xlabel('Fault Types', fontsize=12, fontweight='bold')
        ax.set_ylabel('Number of Files', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xticks(x_positions)
        ax.set_xticklabels(fault_labels, fontsize=11)
        ax.grid(axis='y', alpha=0.3)
        
        # 设置y轴范围
        max_val = max([max(data[ft].values()) if data[ft] else 0 for ft in fault_types])
        ax.set_ylim(0, max_val * 1.15)
        
        # 添加图例
        if present_sizes:
            ax.legend(loc='upper right', fontsize=10)
    
    # 绘制12kHz数据
    plot_frequency_data(ax1, '12k', '12kHz Sampling Frequency')
    
    # 绘制48kHz数据
    plot_frequency_data(ax2, '48k', '48kHz Sampling Frequency')
    
    # 调整整体布局
    plt.suptitle('Bearing Fault Data Distribution by Sampling Frequency and Fault Size', 
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # 保存图片
    plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    print(f"Frequency-Size bar chart saved to: {output_path}")
    
    return fig

def main():
    """主函数"""
    # 设置路径
    current_dir = os.getcwd()
    data_path = os.path.join(current_dir, "..", "处理后数据", "raw_data")
    output_dir = current_dir
    
    print("Bearing Fault Data Analysis - Frequency & Size Distribution")
    print("=" * 60)
    
    try:
        # 按频率和尺寸分类统计
        freq_data = classify_files_by_frequency_and_size(data_path)
        
        # 打印统计结果
        for freq in ['12k', '48k']:
            print(f"\n{freq}Hz Data:")
            for fault_type in ['B', 'OR', 'IR', 'N']:
                if freq_data[freq][fault_type]:
                    print(f"  {fault_type}: {dict(freq_data[freq][fault_type])}")
        
        # 创建柱状图
        chart_path = os.path.join(output_dir, "Frequency_Size_Distribution_Chart.png")
        fig = create_frequency_size_bar_chart(freq_data, chart_path)
        
        print("\nAnalysis completed successfully!")
        
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()