#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强的去噪方法实现
包含小波去噪、EMD去噪、VMD去噪等先进方法
作者: AI Assistant
日期: 2024年9月23日
"""

import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 尝试导入高级去噪库
try:
    import pywt
    PYWT_AVAILABLE = True
    print("✅ PyWavelets 可用")
except ImportError:
    PYWT_AVAILABLE = False
    print("❌ PyWavelets 不可用，请安装: pip install PyWavelets")

try:
    from PyEMD import EEMD, EMD
    PYEMD_AVAILABLE = True
    print("✅ PyEMD 可用")
except ImportError:
    PYEMD_AVAILABLE = False
    print("❌ PyEMD 不可用，请安装: pip install EMD-signal")

try:
    from vmdpy import VMD
    VMD_AVAILABLE = True
    print("✅ VMDpy 可用")
except ImportError:
    VMD_AVAILABLE = False
    print("❌ VMDpy 不可用，请安装: pip install vmdpy")


class EnhancedDenoising:
    """增强的去噪方法类"""
    
    def __init__(self):
        self.fs = 12000  # 默认采样频率
        
    def enhanced_traditional_denoising(self, data, fs=None, rpm=1750):
        """
        增强的传统去噪方法
        包含自适应参数调整和形态学滤波
        """
        if fs is None:
            fs = self.fs
            
        data_flat = data.flatten()
        
        # 1. 自适应高通滤波
        cutoff_hp = max(5, rpm/60 * 0.1)  # 基于转速自适应
        sos_hp = signal.butter(6, cutoff_hp, btype='highpass', fs=fs, output='sos')
        data_filtered = signal.sosfilt(sos_hp, data_flat)
        
        # 2. 自适应低通滤波
        cutoff_lp = min(fs/3, 8000)  # 动态上限
        sos_lp = signal.butter(6, cutoff_lp, btype='lowpass', fs=fs, output='sos')
        data_filtered = signal.sosfilt(sos_lp, data_filtered)
        
        # 3. 扩展的陷波滤波
        notch_freqs = [50, 100, 150, 200, 250, 300]  # 扩展工频谐波
        for freq in notch_freqs:
            if freq < fs/2:
                b_notch, a_notch = signal.iirnotch(freq, 30, fs)
                data_filtered = signal.filtfilt(b_notch, a_notch, data_filtered)
        
        # 4. 改进的中值滤波（多尺度）
        data_filtered = signal.medfilt(data_filtered, kernel_size=3)
        data_filtered = signal.medfilt(data_filtered, kernel_size=5)
        
        # 5. Savitzky-Golay平滑滤波
        if len(data_filtered) > 10:
            data_filtered = signal.savgol_filter(data_filtered, 
                                                min(11, len(data_filtered)//10*2+1), 3)
        
        return data_filtered.reshape(-1, 1)
    
    def wavelet_denoising(self, data, wavelet='db6', threshold_mode='soft', levels=6):
        """
        小波去噪方法
        特别适合轴承振动信号
        """
        if not PYWT_AVAILABLE:
            print("⚠️ PyWavelets不可用，使用传统方法")
            return self.enhanced_traditional_denoising(data)
        
        data_flat = data.flatten()
        
        try:
            # 小波分解
            coeffs = pywt.wavedec(data_flat, wavelet, level=levels)
            
            # 估计噪声标准差（基于最高频细节系数）
            sigma = np.median(np.abs(coeffs[-1])) / 0.6745
            
            # 计算阈值（改进的阈值估计）
            threshold = sigma * np.sqrt(2 * np.log(len(data_flat)))
            
            # 软阈值处理
            coeffs_thresh = list(coeffs)
            coeffs_thresh[1:] = [pywt.threshold(i, threshold, threshold_mode) 
                               for i in coeffs[1:]]
            
            # 重构信号
            data_denoised = pywt.waverec(coeffs_thresh, wavelet)
            
            # 确保长度一致
            if len(data_denoised) != len(data_flat):
                data_denoised = data_denoised[:len(data_flat)]
            
            return data_denoised.reshape(-1, 1)
            
        except Exception as e:
            print(f"⚠️ 小波去噪失败: {e}，使用传统方法")
            return self.enhanced_traditional_denoising(data)
    
    def emd_denoising(self, data, method='EEMD', noise_scale=0.005):
        """
        EMD/EEMD去噪方法
        特别适合非线性和非平稳信号
        """
        if not PYEMD_AVAILABLE:
            print("⚠️ PyEMD不可用，使用小波方法")
            return self.wavelet_denoising(data)
        
        data_flat = data.flatten()
        
        try:
            if method == 'EEMD':
                # 集合经验模态分解
                eemd = EEMD()
                eemd.noise_scale = noise_scale
                eemd.E_noise = 100
                eemd.trials = 100
                imfs = eemd.eemd(data_flat)
            else:
                # 标准EMD
                emd = EMD()
                imfs = emd.emd(data_flat)
            
            if imfs is None or len(imfs) == 0:
                print("⚠️ EMD分解失败，使用小波方法")
                return self.wavelet_denoising(data)
            
            # 计算每个IMF的能量和频率特征
            selected_imfs = []
            for i, imf in enumerate(imfs):
                # 计算IMF的能量比例
                energy_ratio = np.sum(imf**2) / np.sum(data_flat**2)
                
                # 计算IMF的主频
                fft_imf = np.fft.fft(imf)
                freqs = np.fft.fftfreq(len(imf), 1/self.fs)
                main_freq_idx = np.argmax(np.abs(fft_imf[:len(fft_imf)//2]))
                main_freq = abs(freqs[main_freq_idx])
                
                # 选择条件：能量比例 > 0.01 且主频在有效范围内
                if energy_ratio > 0.01 and 1 < main_freq < 5000:
                    selected_imfs.append(imf)
            
            if len(selected_imfs) == 0:
                print("⚠️ 未找到有效IMF，使用小波方法")
                return self.wavelet_denoising(data)
            
            # 重构信号
            data_denoised = np.sum(selected_imfs, axis=0)
            return data_denoised.reshape(-1, 1)
            
        except Exception as e:
            print(f"⚠️ EMD去噪失败: {e}，使用小波方法")
            return self.wavelet_denoising(data)
    
    def vmd_denoising(self, data, K=6, alpha=2000):
        """
        变分模态分解去噪
        对频率分离要求高的信号效果好
        """
        if not VMD_AVAILABLE:
            print("⚠️ VMDpy不可用，使用EMD方法")
            return self.emd_denoising(data)
        
        data_flat = data.flatten()
        
        try:
            # VMD参数设置
            tau = 0.          # 噪声容忍度
            DC = 0            # 无直流分量
            init = 1          # 初始化方式
            tol = 1e-7        # 收敛容忍度
            
            # VMD分解
            u, u_hat, omega = VMD(data_flat, alpha, tau, K, DC, init, tol)
            
            # 基于频率和能量选择模态
            selected_modes = []
            total_energy = np.sum(data_flat**2)
            
            for i in range(K):
                # 计算模态的主频（最后一次迭代的频率）
                main_freq = omega[i][-1] * self.fs / (2 * np.pi)
                
                # 计算模态能量比例
                energy_ratio = np.sum(u[i]**2) / total_energy
                
                # 选择条件：频率在轴承故障频带内且能量比例合适
                if 10 < main_freq < 3000 and energy_ratio > 0.02:
                    selected_modes.append(u[i])
            
            if len(selected_modes) == 0:
                print("⚠️ 未找到有效VMD模态，使用EMD方法")
                return self.emd_denoising(data)
            
            # 重构信号
            data_denoised = np.sum(selected_modes, axis=0)
            return data_denoised.reshape(-1, 1)
            
        except Exception as e:
            print(f"⚠️ VMD去噪失败: {e}，使用EMD方法")
            return self.emd_denoising(data)
    
    def estimate_snr(self, data):
        """
        估计信号的信噪比
        用于自动选择最佳去噪方法
        """
        if PYWT_AVAILABLE:
            try:
                data_flat = data.flatten()
                # 使用小波分解估计噪声
                coeffs = pywt.wavedec(data_flat, 'db4', level=3)
                noise_level = np.median(np.abs(coeffs[-1])) / 0.6745
                signal_power = np.var(data_flat)
                noise_power = noise_level**2
                snr = 10 * np.log10(signal_power / noise_power)
                return max(0, snr)  # 确保非负
            except:
                pass
        
        # 备用SNR估计方法
        data_flat = data.flatten()
        signal_power = np.var(data_flat)
        # 估计噪声为高频成分
        data_smoothed = signal.savgol_filter(data_flat, min(51, len(data_flat)//10*2+1), 3)
        noise = data_flat - data_smoothed
        noise_power = np.var(noise)
        snr = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else 30
        return max(0, snr)
    
    def auto_denoising(self, data, fs=None, rpm=1750):
        """
        自动选择最佳去噪方法
        根据信号质量和可用库自动选择
        """
        if fs is None:
            fs = self.fs
        
        self.fs = fs
        
        print(f"🔍 开始自动去噪分析...")
        
        # 估计信噪比
        snr = self.estimate_snr(data)
        print(f"📊 估计信噪比: {snr:.1f} dB")
        
        # 根据SNR和可用库选择方法
        if snr > 25:  # 高质量信号
            if PYWT_AVAILABLE:
                print("🎯 选择方法: 小波去噪（高质量信号）")
                return self.wavelet_denoising(data, wavelet='db4', levels=4)
            else:
                print("🎯 选择方法: 增强传统去噪")
                return self.enhanced_traditional_denoising(data, fs, rpm)
                
        elif snr > 15:  # 中等质量信号
            if VMD_AVAILABLE:
                print("🎯 选择方法: VMD去噪（中等质量信号）")
                return self.vmd_denoising(data, K=5)
            elif PYWT_AVAILABLE:
                print("🎯 选择方法: 小波去噪（备选）")
                return self.wavelet_denoising(data, wavelet='db6', levels=6)
            else:
                print("🎯 选择方法: 增强传统去噪")
                return self.enhanced_traditional_denoising(data, fs, rpm)
                
        elif snr > 8:  # 低质量信号
            if PYEMD_AVAILABLE:
                print("🎯 选择方法: EEMD去噪（低质量信号）")
                return self.emd_denoising(data, method='EEMD')
            elif PYWT_AVAILABLE:
                print("🎯 选择方法: 小波去噪（备选）")
                return self.wavelet_denoising(data, wavelet='db8', levels=7)
            else:
                print("🎯 选择方法: 增强传统去噪")
                return self.enhanced_traditional_denoising(data, fs, rpm)
                
        else:  # 极低质量信号
            print("🎯 选择方法: 混合去噪（极低质量信号）")
            return self.hybrid_denoising(data, fs, rpm)
    
    def hybrid_denoising(self, data, fs=None, rpm=1750):
        """
        混合去噪方法
        结合多种方法的优势
        """
        if fs is None:
            fs = self.fs
        
        print("🔧 执行混合去噪...")
        
        # Step 1: 传统预处理
        data_pre = self.enhanced_traditional_denoising(data, fs, rpm)
        
        # Step 2: 选择可用的高级方法
        if PYEMD_AVAILABLE:
            print("  Step 2: EEMD处理")
            data_emd = self.emd_denoising(data_pre, method='EEMD')
        else:
            data_emd = data_pre
        
        # Step 3: 小波精细去噪
        if PYWT_AVAILABLE:
            print("  Step 3: 小波精细去噪")
            data_final = self.wavelet_denoising(data_emd, wavelet='db6', levels=5)
        else:
            data_final = data_emd
        
        return data_final
    
    def compare_denoising_methods(self, data, fs=None, rpm=1750, output_dir=None):
        """
        比较不同去噪方法的效果
        生成对比报告和图像
        """
        if fs is None:
            fs = self.fs
        
        self.fs = fs
        data_flat = data.flatten()
        
        print("🔬 开始去噪方法对比...")
        
        # 测试所有可用方法
        methods = {}
        
        # 1. 原始信号
        methods['原始信号'] = data_flat
        
        # 2. 传统方法
        methods['增强传统去噪'] = self.enhanced_traditional_denoising(data, fs, rpm).flatten()
        
        # 3. 小波去噪
        if PYWT_AVAILABLE:
            methods['小波去噪(db4)'] = self.wavelet_denoising(data, 'db4').flatten()
            methods['小波去噪(db6)'] = self.wavelet_denoising(data, 'db6').flatten()
        
        # 4. EMD去噪
        if PYEMD_AVAILABLE:
            methods['EEMD去噪'] = self.emd_denoising(data, 'EEMD').flatten()
        
        # 5. VMD去噪
        if VMD_AVAILABLE:
            methods['VMD去噪'] = self.vmd_denoising(data).flatten()
        
        # 6. 自动选择
        methods['自动选择'] = self.auto_denoising(data, fs, rpm).flatten()
        
        # 计算评估指标
        results = {}
        original_snr = self.estimate_snr(data_flat)
        
        for name, denoised_data in methods.items():
            if name == '原始信号':
                snr = original_snr
                snr_improvement = 0
            else:
                snr = self.estimate_snr(denoised_data)
                snr_improvement = snr - original_snr
            
            # 计算均方根值
            rms = np.sqrt(np.mean(denoised_data**2))
            
            # 计算相对平滑度（变化率的标准差）
            if len(denoised_data) > 1:
                diff = np.diff(denoised_data)
                smoothness = np.std(diff)
            else:
                smoothness = 0
            
            results[name] = {
                'SNR': snr,
                'SNR_Improvement': snr_improvement,
                'RMS': rms,
                'Smoothness': smoothness,
                'Length': len(denoised_data)
            }
        
        # 生成对比图
        if output_dir:
            self._plot_comparison(methods, results, output_dir)
        
        # 打印结果
        print("\n📊 去噪方法对比结果:")
        print("-" * 80)
        print(f"{'方法名称':<12} {'SNR(dB)':<8} {'SNR提升':<8} {'RMS':<10} {'平滑度':<10}")
        print("-" * 80)
        
        for name, metrics in results.items():
            print(f"{name:<12} {metrics['SNR']:<8.1f} {metrics['SNR_Improvement']:<8.1f} "
                  f"{metrics['RMS']:<10.4f} {metrics['Smoothness']:<10.4f}")
        
        # 推荐最佳方法
        best_method = max(results.keys(), 
                         key=lambda x: results[x]['SNR_Improvement'] if x != '原始信号' else -999)
        
        print(f"\n🏆 推荐方法: {best_method}")
        print(f"   SNR改善: {results[best_method]['SNR_Improvement']:.1f} dB")
        
        return results, methods
    
    def _plot_comparison(self, methods, results, output_dir):
        """生成对比图像"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 时域对比图
        fig, axes = plt.subplots(len(methods), 1, figsize=(15, 3*len(methods)))
        if len(methods) == 1:
            axes = [axes]
        
        for i, (name, data) in enumerate(methods.items()):
            time = np.arange(len(data)) / self.fs
            axes[i].plot(time, data, linewidth=0.8)
            axes[i].set_title(f'{name} (SNR: {results[name]["SNR"]:.1f}dB)')
            axes[i].set_ylabel('振幅')
            axes[i].grid(True, alpha=0.3)
            
            if i == len(methods) - 1:
                axes[i].set_xlabel('时间 (秒)')
        
        plt.tight_layout()
        plt.savefig(output_path / 'denoising_comparison_time.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 频域对比图
        fig, axes = plt.subplots(len(methods), 1, figsize=(15, 3*len(methods)))
        if len(methods) == 1:
            axes = [axes]
        
        for i, (name, data) in enumerate(methods.items()):
            # FFT
            fft_data = np.fft.fft(data)
            frequencies = np.fft.fftfreq(len(data), 1/self.fs)
            magnitude = np.abs(fft_data)
            
            # 只显示正频率
            pos_mask = frequencies >= 0
            axes[i].semilogy(frequencies[pos_mask], magnitude[pos_mask])
            axes[i].set_title(f'{name} - 频域')
            axes[i].set_ylabel('幅度')
            axes[i].grid(True, alpha=0.3)
            axes[i].set_xlim(0, min(5000, self.fs/2))
            
            if i == len(methods) - 1:
                axes[i].set_xlabel('频率 (Hz)')
        
        plt.tight_layout()
        plt.savefig(output_path / 'denoising_comparison_frequency.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📁 对比图像已保存到: {output_path}")


def main():
    """主函数 - 演示增强去噪方法"""
    print("🚀 增强去噪方法演示")
    print("=" * 50)
    
    # 检查依赖库
    print("\n📦 依赖库检查:")
    print(f"PyWavelets: {'✅' if PYWT_AVAILABLE else '❌'}")
    print(f"PyEMD: {'✅' if PYEMD_AVAILABLE else '❌'}")
    print(f"VMDpy: {'✅' if VMD_AVAILABLE else '❌'}")
    
    # 创建测试信号
    fs = 12000
    t = np.linspace(0, 2, fs*2)
    
    # 模拟轴承故障信号
    signal_clean = (np.sin(2*np.pi*100*t) + 
                   0.5*np.sin(2*np.pi*200*t) + 
                   0.3*np.sin(2*np.pi*1500*t))  # 基础信号
    
    # 添加故障冲击
    impulse_times = np.arange(0.1, 2, 0.1)  # 10Hz冲击
    for imp_time in impulse_times:
        idx = int(imp_time * fs)
        if idx < len(signal_clean):
            signal_clean[idx:idx+50] += 2 * np.exp(-np.arange(50)/10)
    
    # 添加噪声
    noise = 0.5 * np.random.randn(len(signal_clean))
    signal_noisy = signal_clean + noise
    
    print(f"\n🎵 生成测试信号:")
    print(f"采样频率: {fs} Hz")
    print(f"信号长度: {len(signal_noisy)} 点 ({len(signal_noisy)/fs:.1f} 秒)")
    
    # 创建去噪器
    denoiser = EnhancedDenoising()
    
    # 自动去噪测试
    print(f"\n🔧 自动去噪测试:")
    denoised_auto = denoiser.auto_denoising(signal_noisy, fs, rpm=1750)
    
    # 方法对比
    output_dir = "/Users/gsyoung/MBP_documents/code/Matlab/研究生数学建模/建模实现/处理后数据/denoising_comparison"
    results, methods = denoiser.compare_denoising_methods(
        signal_noisy, fs, rpm=1750, output_dir=output_dir)
    
    print(f"\n✅ 演示完成！")
    print(f"📁 对比结果保存在: {output_dir}")


if __name__ == "__main__":
    main()
