#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据文件分类统计与扇形图生成
"""

import os
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from datetime import datetime

def set_chinese_font():
    """设置中文字体"""
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

def classify_files(data_path):
    """分类统计数据文件"""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"数据路径不存在: {data_path}")
    
    # 获取所有文件夹名称
    all_folders = [f for f in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, f))]
    
    # 过滤掉带有_denoised后缀的文件夹
    original_folders = [f for f in all_folders if not f.endswith('_denoised')]
    
    print(f"总文件夹数: {len(all_folders)}")
    print(f"原始文件夹数（排除denoised）: {len(original_folders)}")
    
    # 分类统计
    categories = {'B': 0, 'OR': 0, 'IR': 0, 'N': 0}
    
    for folder in original_folders:
        folder_upper = folder.upper()
        
        if '_B' in folder_upper:
            categories['B'] += 1
        elif '_OR' in folder_upper:
            categories['OR'] += 1
        elif '_IR' in folder_upper:
            categories['IR'] += 1
        elif '_N' in folder_upper or 'NORMAL' in folder_upper:
            categories['N'] += 1
    
    # 打印分类结果
    print("\n=== 分类统计结果 ===")
    for category, count in categories.items():
        print(f"{category}类: {count} 个文件")
    
    total_classified = sum(categories.values())
    print(f"总计分类文件数: {total_classified}")
    
    return categories

def create_pie_chart(categories, output_path):
    """创建扇形图"""
    set_chinese_font()
    
    # 准备数据
    labels = []
    sizes = []
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    
    category_names = {
        'B': 'B类（球轴承故障）',
        'OR': 'OR类（外圈故障）', 
        'IR': 'IR类（内圈故障）',
        'N': 'N类（正常状态）'
    }
    
    for category, count in categories.items():
        if count > 0:
            labels.append(f"{category_names[category]}\n{count}个")
            sizes.append(count)
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 绘制扇形图
    wedges, texts, autotexts = ax.pie(
        sizes, 
        labels=labels,
        colors=colors[:len(sizes)],
        autopct='%1.1f%%',
        startangle=90,
        textprops={'fontsize': 12}
    )
    
    # 设置标题
    plt.title('轴承故障数据文件类型分布统计', fontsize=16, pad=20)
    
    # 调整百分比文字样式
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(11)
    
    # 确保图形是圆形
    ax.axis('equal')
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"扇形图已保存到: {output_path}")
    
    return fig

def main():
    """主函数"""
    # 设置路径
    current_dir = os.getcwd()
    data_path = os.path.join(current_dir, "..", "处理后数据", "raw_data")
    output_dir = current_dir
    
    print("轴承故障数据文件分析程序")
    print("=" * 50)
    
    try:
        # 分类统计
        categories = classify_files(data_path)
        
        # 创建扇形图
        chart_path = os.path.join(output_dir, "数据类型分布扇形图.png")
        fig = create_pie_chart(categories, chart_path)
        
        print("\n分析完成！")
        
    except Exception as e:
        print(f"程序执行出错: {e}")

if __name__ == "__main__":
    main()