#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
升级您的现有去噪方法演示
展示如何将新的去噪方法集成到现有处理流程中
"""

import numpy as np
import scipy.io
import scipy.signal as signal
from pathlib import Path
from enhanced_denoising_methods import EnhancedDenoising

def demonstrate_upgrade():
    """演示如何升级现有的去噪流程"""
    
    print("🚀 去噪方法升级演示")
    print("=" * 50)
    
    # 1. 加载一个真实的轴承数据文件进行测试
    source_data_path = Path("/Users/gsyoung/MBP_documents/code/Matlab/研究生数学建模/数据集/数据集/源域数据集")
    
    # 查找第一个可用的.mat文件
    mat_files = list(source_data_path.rglob("*.mat"))
    if not mat_files:
        print("❌ 未找到测试数据文件")
        return
    
    test_file = mat_files[0]
    print(f"📁 使用测试文件: {test_file.name}")
    
    # 2. 加载数据
    try:
        data = scipy.io.loadmat(test_file)
        
        # 查找振动信号
        signal_vars = [var for var in data.keys() 
                      if any(x in var.lower() for x in ['de_time', 'fe_time', 'ba_time']) 
                      and isinstance(data[var], np.ndarray)]
        
        if not signal_vars:
            print("❌ 未找到振动信号数据")
            return
        
        # 使用第一个找到的信号
        signal_var = signal_vars[0]
        raw_signal = data[signal_var].flatten()
        
        print(f"📊 信号变量: {signal_var}")
        print(f"📏 信号长度: {len(raw_signal):,} 点")
        
        # 确定采样频率
        fs = 48000 if '48k' in test_file.parent.name else 12000
        print(f"🎵 采样频率: {fs:,} Hz")
        
        # 如果是48kHz，先降采样到12kHz
        if fs == 48000:
            target_fs = 12000
            raw_signal = signal.resample_poly(raw_signal, target_fs, fs)
            fs = target_fs
            print(f"⬇️ 降采样到: {fs:,} Hz")
        
        # 取一段代表性数据（避免处理过长的信号）
        if len(raw_signal) > 60000:
            raw_signal = raw_signal[:60000]
            print(f"📏 截取长度: {len(raw_signal):,} 点 ({len(raw_signal)/fs:.1f}秒)")
        
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return
    
    # 3. 创建增强去噪器
    denoiser = EnhancedDenoising()
    
    print(f"\n🔧 开始去噪方法对比...")
    
    # 4. 对比不同去噪方法
    results, methods = denoiser.compare_denoising_methods(
        raw_signal, 
        fs=fs, 
        rpm=1750,  # 假设转速
        output_dir="/Users/gsyoung/MBP_documents/code/Matlab/研究生数学建模/建模实现/处理后数据/real_data_denoising"
    )
    
    # 5. 生成推荐报告
    print(f"\n📋 升级建议报告:")
    print("=" * 60)
    
    # 找出最佳方法
    best_methods = sorted(
        [(name, metrics['SNR_Improvement']) for name, metrics in results.items() if name != '原始信号'],
        key=lambda x: x[1], reverse=True
    )
    
    print(f"🏆 最佳方法排名:")
    for i, (method, improvement) in enumerate(best_methods[:5], 1):
        stars = "⭐" * min(5, int(improvement/5) + 1)
        print(f"  {i}. {method:<15} SNR提升: {improvement:+6.1f}dB {stars}")
    
    # 6. 具体升级建议
    top_method = best_methods[0][0]
    top_improvement = best_methods[0][1]
    
    print(f"\n💡 具体升级建议:")
    print(f"   当前方法效果已经不错，但可以通过 '{top_method}' 再提升 {top_improvement:.1f}dB")
    
    if top_improvement > 10:
        print(f"   🔥 建议立即升级到 '{top_method}'！")
        print(f"   📈 预期效果：显著提升信号质量和故障检测精度")
    elif top_improvement > 5:
        print(f"   ✅ 建议考虑升级到 '{top_method}'")
        print(f"   📈 预期效果：中等程度提升信号质量")
    else:
        print(f"   💭 当前方法已足够好，升级收益有限")
    
    # 7. 集成建议
    print(f"\n🔧 集成到现有流程的方法:")
    print(f"1. 在 raw_data_processor.py 中替换 apply_denoising 函数")
    print(f"2. 添加智能方法选择：根据信号质量自动选择最佳去噪方法")
    print(f"3. 保留现有方法作为备选：确保向后兼容")
    
    print(f"\n💼 修改代码示例:")
    print(f"```python")
    print(f"# 在您的 raw_data_processor.py 中")
    print(f"from enhanced_denoising_methods import EnhancedDenoising")
    print(f"")
    print(f"def apply_denoising(self, data, fs):")
    print(f"    denoiser = EnhancedDenoising()")
    print(f"    return denoiser.auto_denoising(data, fs)")
    print(f"```")
    
    print(f"\n✅ 演示完成！升级后的去噪效果图像已保存。")

if __name__ == "__main__":
    demonstrate_upgrade()
