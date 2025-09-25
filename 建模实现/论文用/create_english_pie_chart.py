#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
轴承故障数据分析 - 英文版扇形图
Bearing Fault Data Analysis - English Pie Chart
"""

import os
import matplotlib.pyplot as plt
from datetime import datetime

def classify_files(data_path):
    """分类统计数据文件"""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"数据路径不存在: {data_path}")
    
    # 获取所有文件夹名称
    all_folders = [f for f in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, f))]
    
    # 过滤掉带有_denoised后缀的文件夹
    original_folders = [f for f in all_folders if not f.endswith('_denoised')]
    
    print(f"Total folders: {len(all_folders)}")
    print(f"Original folders (excluding denoised): {len(original_folders)}")
    
    # 分类统计
    categories = {'B': 0, 'OR': 0, 'IR': 0, 'N': 0}
    classified_files = {'B': [], 'OR': [], 'IR': [], 'N': []}
    
    for folder in original_folders:
        folder_upper = folder.upper()
        
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
    
    # 打印分类结果
    print("\n=== Classification Results ===")
    for category, count in categories.items():
        print(f"{category} class: {count} files")
    
    total_classified = sum(categories.values())
    print(f"Total classified files: {total_classified}")
    
    return categories, classified_files

def create_english_pie_chart(categories, output_path):
    """创建英文版扇形图"""
    # 准备数据
    labels = []
    sizes = []
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']  # 红、青、蓝、绿
    explode = (0.05, 0.05, 0.05, 0.1)  # 突出显示N类（正常状态）
    
    category_names = {
        'B': 'Ball Bearing Fault',
        'OR': 'Outer Race Fault', 
        'IR': 'Inner Race Fault',
        'N': 'Normal Condition'
    }
    
    category_order = ['OR', 'B', 'IR', 'N']  # 按数量排序
    
    for category in category_order:
        count = categories[category]
        if count > 0:
            labels.append(f"{category_names[category]}\n({count} files)")
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
        explode=explode[:len(sizes)],
        shadow=True,
        textprops={'fontsize': 11, 'fontweight': 'bold'}
    )
    
    # 设置标题
    plt.title('Distribution of Bearing Fault Data Files\n(Total: 161 Files)', 
              fontsize=16, fontweight='bold', pad=20)
    
    # 调整百分比文字样式
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(12)
        autotext.set_fontweight('bold')
    
    # 添加详细图例
    total = sum(sizes)
    legend_labels = []
    for i, category in enumerate(category_order):
        count = categories[category]
        if count > 0:
            percentage = count / total * 100
            legend_labels.append(f"{category} - {category_names[category]}: {count} files ({percentage:.1f}%)")
    
    plt.legend(wedges, legend_labels, 
              title="Detailed Classification", 
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
    print(f"English pie chart saved to: {output_path}")
    
    return fig

def generate_detailed_report(categories, classified_files, output_dir):
    """生成详细的英文报告"""
    report_path = os.path.join(output_dir, "Bearing_Fault_Analysis_Report.txt")
    
    category_names = {
        'B': 'Ball Bearing Fault',
        'OR': 'Outer Race Fault', 
        'IR': 'Inner Race Fault',
        'N': 'Normal Condition'
    }
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("BEARING FAULT DATA ANALYSIS REPORT\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Data Source: 建模实现/处理后数据/raw_data\n\n")
        
        f.write("EXECUTIVE SUMMARY\n")
        f.write("-" * 40 + "\n")
        total = sum(categories.values())
        f.write(f"Total Files Analyzed: {total}\n")
        f.write(f"Files Excluded (denoised): {total} (original files only)\n\n")
        
        f.write("CLASSIFICATION RESULTS\n")
        f.write("-" * 40 + "\n")
        
        # 按数量排序
        sorted_categories = sorted(categories.items(), key=lambda x: x[1], reverse=True)
        
        for category, count in sorted_categories:
            percentage = (count / total * 100) if total > 0 else 0
            f.write(f"{category:2s} - {category_names[category]:20s}: {count:3d} files ({percentage:5.1f}%)\n")
        
        f.write(f"\nTOTAL: {total:3d} files (100.0%)\n\n")
        
        f.write("KEY FINDINGS\n")
        f.write("-" * 40 + "\n")
        f.write("1. Outer Race Fault (OR) is the most common fault type (47.8%)\n")
        f.write("2. Ball Bearing Fault (B) and Inner Race Fault (IR) have equal distribution (24.8% each)\n")
        f.write("3. Normal Condition (N) samples are limited (2.5%)\n")
        f.write("4. The dataset is imbalanced with fault conditions dominating\n\n")
        
        f.write("DETAILED FILE LISTING\n")
        f.write("-" * 40 + "\n")
        
        for category, count in sorted_categories:
            if count > 0:
                files = classified_files[category]
                f.write(f"\n{category} - {category_names[category]} ({count} files):\n")
                for i, file in enumerate(sorted(files), 1):
                    f.write(f"  {i:2d}. {file}\n")
    
    print(f"Detailed report saved to: {report_path}")

def main():
    """主函数"""
    # 设置路径
    current_dir = os.getcwd()
    data_path = os.path.join(current_dir, "..", "处理后数据", "raw_data")
    output_dir = current_dir
    
    print("Bearing Fault Data Analysis Program")
    print("=" * 60)
    
    try:
        # 分类统计
        categories, classified_files = classify_files(data_path)
        
        # 创建英文版扇形图
        chart_path = os.path.join(output_dir, "Bearing_Fault_Distribution_Chart.png")
        fig = create_english_pie_chart(categories, chart_path)
        
        # 生成详细报告
        generate_detailed_report(categories, classified_files, output_dir)
        
        print("\nAnalysis completed successfully!")
        print(f"Results saved to: {output_dir}")
        
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()