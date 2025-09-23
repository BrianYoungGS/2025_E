#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速开始滤波指南 - 30秒上手高效滤波
最简单的使用方式，立即获得显著的滤波效果提升

作者: AI Assistant
日期: 2024年9月23日
版本: v1.0 - 快速入门版
"""

import numpy as np
import scipy.io
from pathlib import Path

# 导入统一滤波工具包
try:
    from unified_filtering_toolkit import UnifiedFilteringToolkit
    print("✅ 高级滤波工具包加载成功")
except ImportError:
    print("❌ 请确保 unified_filtering_toolkit.py 在当前目录")
    exit(1)


def quick_demo():
    """30秒快速演示"""
    print("🚀 30秒快速滤波演示")
    print("=" * 40)
    
    # 1. 创建工具包（一行代码）
    toolkit = UnifiedFilteringToolkit()
    print("✅ 步骤1: 创建滤波工具包")
    
    # 2. 生成测试信号（模拟轴承数据）
    fs = 12000  # 采样频率
    t = np.linspace(0, 1, fs)
    
    # 模拟含噪声的轴承故障信号
    clean_signal = np.sin(2*np.pi*157*t) + 0.5*np.sin(2*np.pi*314*t)  # 故障频率
    noise = 0.3 * np.random.randn(len(clean_signal))
    noisy_signal = clean_signal + noise
    
    print("✅ 步骤2: 准备测试数据")
    
    # 3. 智能滤波（一行代码）
    filtered_signal = toolkit.filter(noisy_signal, fs, method='auto')
    print("✅ 步骤3: 智能滤波完成")
    
    # 4. 显示效果
    original_snr = toolkit._estimate_snr(noisy_signal)
    filtered_snr = toolkit._estimate_snr(filtered_signal)
    improvement = filtered_snr - original_snr
    
    print(f"\n📊 滤波效果:")
    print(f"  原始信号SNR:  {original_snr:.1f} dB")
    print(f"  滤波后SNR:   {filtered_snr:.1f} dB")
    print(f"  🎯 效果提升:   {improvement:.1f} dB")
    
    return improvement


def real_data_example():
    """真实数据示例"""
    print("\n🔧 真实轴承数据滤波示例")
    print("=" * 40)
    
    # 查找真实的轴承数据文件
    data_dir = Path("/Users/gsyoung/MBP_documents/code/Matlab/研究生数学建模/数据集/数据集/源域数据集")
    
    # 寻找 .mat 文件
    mat_files = list(data_dir.rglob("*.mat"))
    
    if not mat_files:
        print("❌ 未找到轴承数据文件，使用模拟数据")
        return quick_demo()
    
    # 使用第一个找到的文件
    test_file = mat_files[0]
    print(f"📁 使用文件: {test_file.name}")
    
    try:
        # 加载真实数据
        data = scipy.io.loadmat(test_file)
        
        # 查找振动信号变量
        signal_vars = [var for var in data.keys() 
                      if any(x in var.lower() for x in ['de_time', 'fe_time', 'ba_time'])
                      and isinstance(data[var], np.ndarray)]
        
        if not signal_vars:
            print("❌ 未找到振动信号变量，使用模拟数据")
            return quick_demo()
        
        # 使用第一个找到的信号
        signal_var = signal_vars[0]
        raw_signal = data[signal_var].flatten()
        
        # 确定采样频率
        fs = 48000 if '48k' in test_file.parent.name else 12000
        
        # 如果数据太长，截取一段
        if len(raw_signal) > 60000:
            raw_signal = raw_signal[:60000]
        
        print(f"📊 数据信息: {len(raw_signal)} 点, {fs} Hz")
        print(f"🎵 信号变量: {signal_var}")
        
        # 创建滤波工具包
        toolkit = UnifiedFilteringToolkit()
        
        # 智能滤波
        print("🔧 开始智能滤波...")
        filtered_signal = toolkit.filter(raw_signal, fs, method='auto')
        
        # 计算效果
        improvement = toolkit._calculate_snr_improvement(raw_signal, filtered_signal)
        
        print(f"\n📊 真实数据滤波效果:")
        print(f"  🎯 SNR提升: {improvement:.1f} dB")
        print(f"  📏 数据长度: {len(filtered_signal):,} 点")
        
        if improvement > 5:
            print("  🚀 效果: 显著提升！")
        elif improvement > 2:
            print("  ✅ 效果: 有效改善")
        else:
            print("  💭 效果: 轻微改善")
        
        return improvement
        
    except Exception as e:
        print(f"❌ 处理真实数据失败: {e}")
        print("🔄 改用模拟数据演示")
        return quick_demo()


def integration_example():
    """集成到现有代码的示例"""
    print("\n🔧 集成到现有代码示例")
    print("=" * 40)
    
    print("方法1: 直接替换现有滤波函数")
    print("""
def apply_denoising(self, data, fs):
    # 原来的复杂滤波代码...
    
    # 替换为一行代码：
    from unified_filtering_toolkit import UnifiedFilteringToolkit
    toolkit = UnifiedFilteringToolkit()
    return toolkit.filter(data, fs, method='auto')
    """)
    
    print("方法2: 保守升级（推荐）")
    print("""
def apply_denoising(self, data, fs, use_advanced=True):
    if use_advanced:
        try:
            from unified_filtering_toolkit import UnifiedFilteringToolkit
            toolkit = UnifiedFilteringToolkit()
            return toolkit.filter(data, fs, method='auto')
        except Exception as e:
            print(f"高级滤波失败: {e}, 使用传统方法")
            return self.traditional_filter(data, fs)
    else:
        return self.traditional_filter(data, fs)
    """)
    
    print("方法3: 批量处理")
    print("""
# 批量处理多个数据文件
toolkit = UnifiedFilteringToolkit()
data_list = [load_data(file) for file in data_files]
filtered_list = toolkit.batch_filter(data_list, fs=12000, method='auto')
    """)


def performance_comparison():
    """性能对比演示"""
    print("\n📊 性能对比演示")
    print("=" * 40)
    
    # 创建工具包
    toolkit = UnifiedFilteringToolkit()
    
    # 生成测试信号
    fs = 12000
    t = np.linspace(0, 1, fs)
    signal_data = np.sin(2*np.pi*157*t) + 0.3*np.random.randn(len(t))
    
    # 对比不同方法
    methods = {
        'fast': '快速滤波（实时应用）',
        'quality': '高质量滤波（离线分析）',
        'auto': '智能自动选择'
    }
    
    print("🧪 测试不同滤波方法:")
    
    for method, description in methods.items():
        try:
            import time
            start_time = time.time()
            
            filtered = toolkit.filter(signal_data, fs, method=method)
            
            process_time = time.time() - start_time
            snr_improvement = toolkit._calculate_snr_improvement(signal_data, filtered)
            
            print(f"  {method:8s}: {description}")
            print(f"           SNR提升 {snr_improvement:5.1f}dB, 时间 {process_time:.3f}s")
            
        except Exception as e:
            print(f"  {method:8s}: 失败 ({e})")
    
    print(f"\n💡 建议:")
    print(f"  - 实时系统使用 'fast'")
    print(f"  - 离线分析使用 'quality'")
    print(f"  - 不确定时使用 'auto'")


def main():
    """主函数 - 完整演示"""
    print("🎉 轴承振动信号高效滤波 - 快速入门指南")
    print("=" * 60)
    
    # 1. 快速演示
    demo_improvement = quick_demo()
    
    # 2. 真实数据测试
    real_improvement = real_data_example()
    
    # 3. 性能对比
    performance_comparison()
    
    # 4. 集成示例
    integration_example()
    
    # 5. 总结
    print(f"\n🎯 总结:")
    print(f"  📈 演示数据提升: {demo_improvement:.1f} dB")
    print(f"  📈 真实数据提升: {real_improvement:.1f} dB")
    print(f"  ⚡ 使用方式: 一行代码即可")
    print(f"  🎛️ 方法选择: 'auto' 智能选择")
    print(f"  🔧 集成方式: 直接替换现有函数")
    
    print(f"\n🚀 立即开始使用:")
    print(f"```python")
    print(f"from unified_filtering_toolkit import UnifiedFilteringToolkit")
    print(f"")
    print(f"toolkit = UnifiedFilteringToolkit()")
    print(f"filtered_data = toolkit.filter(your_data, fs=12000, method='auto')")
    print(f"```")
    
    print(f"\n✅ 快速入门完成！您已掌握高效滤波的核心用法。")


if __name__ == "__main__":
    main()
