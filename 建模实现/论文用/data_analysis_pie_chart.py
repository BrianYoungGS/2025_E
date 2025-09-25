#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据文件分类统计与扇形图生成
分析raw_data文件夹中的数据文件，按照B、OR、IR、N四个类别进行统计
忽略带有_denoised后缀的文件
"""

import os
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from collections import Counter
import numpy as np
from datetime import datetime

def get_available_chinese_fonts():
    """获取系统中可用的中文字体"""
    chinese_fonts = []
    for font in fm.fontManager.ttflist:
        font_name = font.name.lower()
        if any(keyword in font_name for keyword in ['simhei', 'simsun', 'microsoft', 'dengxian', 'kaiti', 'fangsong']):
            chinese_fonts.append(font.name)
    return chinese_fonts

def set_chinese_font():
    """设置中文字体"""
    # 尝试常见的中文字体
    chinese_fonts = ['SimHei', 'Microsoft YaHei', 'SimSun', 'DejaVu Sans', 'Arial Unicode MS']
    
    for font in chinese_fonts:
        try:
            plt.rcParams['font.sans-serif'] = [font]
            plt.rcParams['axes.unicode_minus'] = False
            return font
        except:
            continue
    
    # 如果都不可用，使用系统默认字体
    available_fonts = get_available_chinese_fonts()
    if available_fonts:
        plt.rcParams['font.sans-serif'] = [available_fonts[0]]
        plt.rcParams['axes.unicode_minus'] = False
        return available_fonts[0]
    
    print("警告: 未找到合适的中文字体，可能无法正确显示中文")
    return "default"

def classify_files(data_path):
    """
    分类统计数据文件
    
    Args:
        data_path: raw_data文件夹路径
    
    Returns:
        dict: 各类别的文件数量统计
    """
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
    classified_files = {'B': [], 'OR': [], 'IR': [], 'N': []}
    unclassified = []
    
    for folder in original_folders:
        folder_upper = folder.upper()
        
        # 分类逻辑
        if '_B' in folder_upper:
            categories['B'] += 1
            classified_files['B'].append(folder)
        elif '_OR' in folder_upper:
            categories['OR'] += 1
            classified_files['OR'].append(folder)
        elif '_IR' in folder_upper:
            categories['IR'] += 1
            classified_files['IR'].append(folder)
        elif '_N' in folder_upper or 'NORMAL' in folder_upper:
            categories['N'] += 1
            classified_files['N'].append(folder)
        else:
            unclassified.append(folder)
    
    # 打印分类结果
    print("\n=== 分类统计结果 ===")
    for category, count in categories.items():
        print(f"{category}类: {count} 个文件")
    
    if unclassified:
        print(f"\n未分类文件: {len(unclassified)} 个")
        for file in unclassified[:5]:  # 只显示前5个
            print(f"  - {file}")
        if len(unclassified) > 5:
            print(f"  ... 还有 {len(unclassified) - 5} 个")
    
    total_classified = sum(categories.values())
    print(f"\n总计分类文件数: {total_classified}")
    
    return categories, classified_files, original_folders

def create_pie_chart(categories, output_path):
    """
    创建扇形图
    
    Args:
        categories: 各类别统计数据
        output_path: 输出文件路径
    """
    # 设置中文字体
    font_name = set_chinese_font()
    print(f"使用字体: {font_name}")
    
    # 准备数据
    labels = []
    sizes = []
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']  # 红、青、蓝、绿
    
    category_names = {
        'B': 'B类（球轴承故障）',
        'OR': 'OR类（外圈故障）', 
        'IR': 'IR类（内圈故障）',
        'N': 'N类（正常状态）'
    }
    
    for category, count in categories.items():
        if count > 0:  # 只显示有数据的类别
            labels.append(f"{category_names[category]}\n{count}个")
            sizes.append(count)
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 绘制扇形图
    wedges, texts, autotexts = ax.pie(
        sizes, 
        labels=labels,
        colors=colors[:len(sizes)],
        autopct='%1.1f%%',
        startangle=90,
        textprops={'fontsize': 12, 'fontweight': 'bold'}
    )
    
    # 设置标题
    plt.title('轴承故障数据文件类型分布统计', fontsize=16, fontweight='bold', pad=20)
    
    # 调整百分比文字样式
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(11)
        autotext.set_fontweight('bold')
    
    # 添加图例
    legend_labels = [f"{category_names[cat]} ({count}个, {count/sum(sizes)*100:.1f}%)" 
                    for cat, count in categories.items() if count > 0]
    
    plt.legend(wedges, legend_labels, 
              title="数据类别详情", 
              loc="center left", 
              bbox_to_anchor=(1, 0, 0.5, 1),
              fontsize=10)
    
    # 确保图形是圆形
    ax.axis('equal')
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    print(f"扇形图已保存到: {output_path}")
    
    # 显示图片
    plt.show()
    
    return fig

def generate_report(categories, classified_files, original_folders, output_dir):
    """生成详细报告"""
    report_path = os.path.join(output_dir, "数据分析报告.txt")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("轴承故障数据文件分析报告\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"分析时间: {plt.datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"数据源路径: 建模实现/处理后数据/raw_data\n\n")
        
        f.write("统计概要:\n")
        f.write("-" * 30 + "\n")
        total = sum(categories.values())
        f.write(f"总文件数: {total}\n")
        
        for category, count in categories.items():
            percentage = (count / total * 100) if total > 0 else 0
            category_names = {
                'B': 'B类（球轴承故障）',
                'OR': 'OR类（外圈故障）', 
                'IR': 'IR类（内圈故障）',
                'N': 'N类（正常状态）'
            }
            f.write(f"{category_names[category]}: {count}个 ({percentage:.1f}%)\n")
        
        f.write("\n详细文件列表:\n")
        f.write("-" * 30 + "\n")
        
        for category, files in classified_files.items():
            if files:
                category_names = {
                    'B': 'B类（球轴承故障）',
                    'OR': 'OR类（外圈故障）', 
                    'IR': 'IR类（内圈故障）',
                    'N': 'N类（正常状态）'
                }
                f.write(f"\n{category_names[category]} ({len(files)}个):\n")
                for i, file in enumerate(files, 1):
                    f.write(f"  {i:2d}. {file}\n")
    
    print(f"详细报告已保存到: {report_path}")

def main():
    """主函数"""
    # 设置路径
    current_dir = os.getcwd()
    data_path = os.path.join(current_dir, "建模实现", "处理后数据", "raw_data")
    output_dir = os.path.join(current_dir, "建模实现", "论文用")
    
    print("轴承故障数据文件分析程序")
    print("=" * 50)
    print(f"数据路径: {data_path}")
    print(f"输出路径: {output_dir}")
    
    try:
        # 分类统计
        categories, classified_files, original_folders = classify_files(data_path)
        
        # 创建扇形图
        chart_path = os.path.join(output_dir, "数据类型分布扇形图.png")
        fig = create_pie_chart(categories, chart_path)
        
        # 生成详细报告
        generate_report(categories, classified_files, original_folders, output_dir)
        
        print("\n分析完成！")
        print(f"结果已保存到文件夹: {output_dir}")
        
    except Exception as e:
        print(f"程序执行出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()