#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
原始数据处理器 - 生成完整的原始数据和去噪数据
目标：161个原始文件 → 322个完整数据文件（161个原始 + 161个去噪）
"""

import os
import numpy as np
import scipy.io as sio
import scipy.signal as signal
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import json
from datetime import datetime

class RawDataProcessor:
    """原始数据处理器 - 保留完整数据不做分段"""
    
    def __init__(self, source_base_path, output_base_path):
        self.source_base = Path(source_base_path)
        self.output_base = Path(output_base_path)
        self.processed_count = 0
        
        # 轴承参数
        self.bearing_params = {
            'SKF6205': {'n': 9, 'd': 0.3126, 'D': 1.537},  # DE轴承
            'SKF6203': {'n': 9, 'd': 0.2656, 'D': 1.122}   # FE轴承
        }
    
    def load_mat_file(self, file_path):
        """加载.mat文件"""
        try:
            return sio.loadmat(str(file_path))
        except Exception as e:
            print(f"❌ 加载文件失败 {file_path}: {e}")
            return None
    
    def downsample_signal(self, data, original_fs, target_fs):
        """降采样信号"""
        if original_fs == target_fs:
            return data
        
        # 计算降采样比例
        downsample_factor = int(original_fs / target_fs)
        
        # 使用scipy的decimate函数进行降采样
        downsampled = signal.decimate(data.flatten(), downsample_factor, ftype='iir')
        
        return downsampled.reshape(-1, 1)
    
    def apply_denoising(self, data, fs):
        """应用去噪滤波器"""
        data_flat = data.flatten()
        
        # 1. 高通滤波 (10Hz)
        sos_hp = signal.butter(4, 10, btype='highpass', fs=fs, output='sos')
        data_filtered = signal.sosfilt(sos_hp, data_flat)
        
        # 2. 低通滤波 (5000Hz)
        sos_lp = signal.butter(4, 5000, btype='lowpass', fs=fs, output='sos')
        data_filtered = signal.sosfilt(sos_lp, data_filtered)
        
        # 3. 陷波滤波 (50Hz工频及其谐波)
        for freq in [50, 100, 150]:
            if freq < fs/2:
                b_notch, a_notch = signal.iirnotch(freq, 30, fs)
                data_filtered = signal.filtfilt(b_notch, a_notch, data_filtered)
        
        # 4. 中值滤波去除脉冲噪声
        data_filtered = signal.medfilt(data_filtered, kernel_size=3)
        
        return data_filtered.reshape(-1, 1)
    
    def extract_time_features(self, data):
        """提取时域特征"""
        data_flat = data.flatten()
        
        features = {
            'P1_Mean': float(np.mean(data_flat)),
            'P2_RMS': float(np.sqrt(np.mean(data_flat**2))),
            'P3_Variance': float(np.var(data_flat)),
            'P4_Std': float(np.std(data_flat)),
            'P5_Skewness': float(self._skewness(data_flat)),
            'P6_Kurtosis': float(self._kurtosis(data_flat)),
            'P7_Peak': float(np.max(np.abs(data_flat))),
            'P8_Peak2Peak': float(np.max(data_flat) - np.min(data_flat)),
            'P9_CrestFactor': float(np.max(np.abs(data_flat)) / np.sqrt(np.mean(data_flat**2))),
            'P10_ShapeFactor': float(np.sqrt(np.mean(data_flat**2)) / np.mean(np.abs(data_flat))),
            'P11_ImpulseFactor': float(np.max(np.abs(data_flat)) / np.mean(np.abs(data_flat))),
            'P12_MarginFactor': float(np.max(np.abs(data_flat)) / (np.mean(np.sqrt(np.abs(data_flat))))**2),
            'P13_Energy': float(np.sum(data_flat**2)),
            'P14_Entropy': float(self._calculate_entropy(data_flat)),
            'P15_ZeroCrossing': int(np.sum(np.diff(np.sign(data_flat)) != 0)),
            'P16_MeanFreq': float(self._mean_frequency(data_flat))
        }
        
        return features
    
    def compute_fft_features(self, data, fs):
        """计算FFT特征"""
        data_flat = data.flatten()
        
        # 计算FFT
        fft_data = np.fft.fft(data_flat)
        fft_magnitude = np.abs(fft_data[:len(fft_data)//2])
        frequencies = np.fft.fftfreq(len(data_flat), 1/fs)[:len(fft_data)//2]
        
        # 频域特征
        freq_mean = float(np.sum(frequencies * fft_magnitude) / np.sum(fft_magnitude))
        freq_var = float(np.sum((frequencies - freq_mean)**2 * fft_magnitude) / np.sum(fft_magnitude))
        
        features = {
            'P17_FreqMean': freq_mean,
            'P18_FreqVar': freq_var,
            'P19_FreqStd': float(np.sqrt(freq_var)),
            'P20_FreqSkew': float(self._freq_skewness(frequencies, fft_magnitude)),
            'P21_FreqKurt': float(self._freq_kurtosis(frequencies, fft_magnitude)),
            'P22_FreqRMS': float(np.sqrt(np.mean(fft_magnitude**2))),
            'P23_SpectralCentroid': float(np.sum(frequencies * fft_magnitude) / np.sum(fft_magnitude)),
            'P24_SpectralRolloff': float(self._spectral_rolloff(frequencies, fft_magnitude)),
            'P25_SpectralFlatness': float(self._spectral_flatness(fft_magnitude)),
            'P26_SpectralBandwidth': float(self._spectral_bandwidth(frequencies, fft_magnitude)),
            'P27_DominantFreq': float(frequencies[np.argmax(fft_magnitude)]),
            'P28_FreqPeak': float(np.max(fft_magnitude)),
            'P29_TotalPower': float(np.sum(fft_magnitude**2))
        }
        
        return features, fft_magnitude, frequencies
    
    def calculate_theoretical_fault_frequencies(self, rpm, bearing_type):
        """计算理论故障频率"""
        fr = rpm / 60  # 转频(Hz)
        
        if bearing_type not in self.bearing_params:
            return {}
        
        params = self.bearing_params[bearing_type]
        n, d, D = params['n'], params['d'], params['D']
        
        # 计算特征频率
        bpfo = (n * fr / 2) * (1 - d * np.cos(0) / D)  # 外圈故障频率
        bpfi = (n * fr / 2) * (1 + d * np.cos(0) / D)  # 内圈故障频率
        bsf = (D * fr / (2 * d)) * (1 - (d * np.cos(0) / D)**2)  # 滚动体故障频率
        ftf = (fr / 2) * (1 - d * np.cos(0) / D)  # 保持架故障频率
        
        return {
            'BPFO': float(bpfo),
            'BPFI': float(bpfi),
            'BSF': float(bsf),
            'FTF': float(ftf),
            'FR': float(fr)
        }
    
    def analyze_frequency_components(self, fft_magnitude, frequencies, fs, rpm, bearing_type):
        """分析频率成分"""
        # 主频检测
        peak_indices, _ = signal.find_peaks(fft_magnitude, height=np.max(fft_magnitude)*0.1)
        
        if len(peak_indices) > 0:
            main_peak_idx = peak_indices[np.argmax(fft_magnitude[peak_indices])]
            main_frequency = frequencies[main_peak_idx]
            main_amplitude = fft_magnitude[main_peak_idx]
        else:
            main_frequency = frequencies[np.argmax(fft_magnitude)]
            main_amplitude = np.max(fft_magnitude)
        
        # 谐波检测
        harmonics = []
        for h in range(2, 6):  # 检测2-5次谐波
            harmonic_freq = main_frequency * h
            if harmonic_freq < fs/2:
                # 查找谐波附近的峰值
                freq_range = 5  # Hz
                mask = (frequencies >= harmonic_freq - freq_range) & (frequencies <= harmonic_freq + freq_range)
                if np.any(mask):
                    harmonic_idx = np.argmax(fft_magnitude[mask])
                    actual_idx = np.where(mask)[0][harmonic_idx]
                    harmonics.append({
                        'order': h,
                        'frequency': float(frequencies[actual_idx]),
                        'amplitude': float(fft_magnitude[actual_idx])
                    })
        
        # 理论故障频率
        theoretical_freqs = self.calculate_theoretical_fault_frequencies(rpm, bearing_type)
        
        analysis_result = {
            'main_frequency': float(main_frequency),
            'main_amplitude': float(main_amplitude),
            'harmonics': harmonics,
            'theoretical_frequencies': theoretical_freqs,
            'peak_count': len(peak_indices)
        }
        
        return analysis_result
    
    def plot_time_domain(self, data, fs, output_path, data_name, data_type):
        """绘制时域信号"""
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        
        data_flat = data.flatten()
        time = np.arange(len(data_flat)) / fs
        
        fig, ax = plt.subplots(figsize=(15, 6))
        ax.plot(time, data_flat, 'b-', linewidth=0.8)
        ax.set_xlabel('时间 (秒)')
        ax.set_ylabel('振幅')
        ax.set_title(f'{data_name} - {data_type} - 时域信号')
        ax.grid(True, alpha=0.3)
        
        # 添加统计信息
        rms = np.sqrt(np.mean(data_flat**2))
        peak = np.max(np.abs(data_flat))
        peak2peak = np.max(data_flat) - np.min(data_flat)
        
        info_text = f'类型: {data_type}\nRMS: {rms:.4f}\nPeak: {peak:.4f}\nP-P: {peak2peak:.4f}\n点数: {len(data_flat):,}\n时长: {len(data_flat)/fs:.1f}秒'
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_frequency_domain(self, fft_magnitude, frequencies, output_path, data_name, data_type, freq_analysis):
        """绘制频域信号"""
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        
        fig, ax = plt.subplots(figsize=(15, 6))
        ax.plot(frequencies, fft_magnitude, 'b-', linewidth=0.8)
        ax.set_xlabel('频率 (Hz)')
        ax.set_ylabel('幅值')
        ax.set_title(f'{data_name} - {data_type} - 频域信号')
        ax.grid(True, alpha=0.3)
        
        # 标注主频
        main_freq = freq_analysis['main_frequency']
        main_amp = freq_analysis['main_amplitude']
        ax.plot(main_freq, main_amp, 'ro', markersize=8)
        ax.annotate(f'主频: {main_freq:.1f}Hz', 
                   xy=(main_freq, main_amp), xytext=(main_freq+50, main_amp),
                   arrowprops=dict(arrowstyle='->', color='red'))
        
        # 标注谐波
        for harmonic in freq_analysis['harmonics'][:3]:  # 只标注前3个谐波
            h_freq = harmonic['frequency']
            h_amp = harmonic['amplitude']
            ax.plot(h_freq, h_amp, 'go', markersize=6)
            ax.annotate(f'{harmonic["order"]}次谐波', 
                       xy=(h_freq, h_amp), xytext=(h_freq+30, h_amp*0.8),
                       arrowprops=dict(arrowstyle='->', color='green'))
        
        # 标注理论故障频率
        theoretical = freq_analysis['theoretical_frequencies']
        colors = {'BPFO': 'orange', 'BPFI': 'purple', 'BSF': 'brown', 'FTF': 'pink'}
        for fault_type, freq_val in theoretical.items():
            if fault_type != 'FR' and freq_val < max(frequencies):
                ax.axvline(x=freq_val, color=colors.get(fault_type, 'gray'), 
                          linestyle='--', alpha=0.7, label=f'{fault_type}: {freq_val:.1f}Hz')
        
        ax.legend()
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def process_single_file(self, file_path, category, file_index):
        """
        处理单个文件，生成原始数据和去噪数据两个版本
        """
        print(f"\n📄 处理文件 {file_index+1}/161: {file_path.name}")
        
        # 加载数据
        mat_data = self.load_mat_file(file_path)
        if mat_data is None:
            return []
        
        # 确定采样频率和降采样策略
        if '48kHz' in category:
            original_fs = 48000
            target_fs = 12000
            need_downsample = True
        else:
            original_fs = 12000
            target_fs = 12000
            need_downsample = False
        
        # 查找数据变量
        time_vars = [k for k in mat_data.keys() if not k.startswith('__') and 'time' in k.lower()]
        rpm_vars = [k for k in mat_data.keys() if not k.startswith('__') and 'rpm' in k.lower()]
        
        if not time_vars:
            print(f"❌ 未找到时域数据变量")
            return []
        
        # 获取RPM
        rpm = 1760  # 默认转速
        if rpm_vars:
            rpm_data = mat_data[rpm_vars[0]]
            if isinstance(rpm_data, np.ndarray):
                rpm = float(rpm_data.flatten()[0])
        
        processed_files = []
        
        # 选择主要的时域变量（通常是DE_time）
        primary_var = time_vars[0]
        for var_name in time_vars:
            if 'DE_time' in var_name:
                primary_var = var_name
                break
        
        data = mat_data[primary_var]
        if not isinstance(data, np.ndarray) or len(data.shape) < 1:
            print(f"❌ 数据格式错误: {primary_var}")
            return []
        
        print(f"  处理变量: {primary_var}, 长度: {len(data):,}")
        
        # 降采样（如果需要）
        if need_downsample:
            data_downsampled = self.downsample_signal(data, original_fs, target_fs)
            print(f"  降采样后长度: {len(data_downsampled):,}")
        else:
            data_downsampled = data
        
        # 确定故障类型
        fault_type = self._determine_fault_type(file_path.name, category)
        
        # 确定轴承类型
        bearing_type = 'SKF6205' if 'DE' in primary_var else 'SKF6203'
        
        # 生成文件名（加上类别前缀避免重名）
        category_prefix = category.replace('_data', '').replace('kHz_', 'k_')
        base_name = f"{category_prefix}_{file_path.stem}"
        
        # 1. 处理原始数据（降采样后，但未去噪）
        print(f"    📁 生成原始数据文件...")
        
        # 提取特征
        raw_time_features = self.extract_time_features(data_downsampled)
        raw_freq_features, raw_fft_magnitude, raw_frequencies = self.compute_fft_features(data_downsampled, target_fs)
        raw_freq_analysis = self.analyze_frequency_components(raw_fft_magnitude, raw_frequencies, target_fs, rpm, bearing_type)
        
        # 创建原始数据文件夹
        raw_data_dir = self.output_base / f"{base_name}"
        raw_data_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存原始数据
        np.save(raw_data_dir / f"{base_name}_raw_data.npy", data_downsampled)
        
        # 绘制原始数据图像
        self.plot_time_domain(data_downsampled, target_fs, 
                            raw_data_dir / f"{base_name}_time_domain.png", 
                            base_name, "原始数据")
        self.plot_frequency_domain(raw_fft_magnitude, raw_frequencies, 
                                 raw_data_dir / f"{base_name}_frequency_domain.png", 
                                 base_name, "原始数据", raw_freq_analysis)
        
        # 保存原始数据特征
        raw_all_features = {**raw_time_features, **raw_freq_features, 'fault_type': fault_type, 'data_type': 'raw'}
        raw_features_df = pd.DataFrame([raw_all_features])
        raw_features_df.to_csv(raw_data_dir / f"{base_name}_features.csv", index=False, encoding='utf-8-sig')
        
        # 保存原始数据频率分析
        raw_freq_analysis['data_type'] = 'raw'
        raw_freq_df = pd.DataFrame([raw_freq_analysis])
        raw_freq_df.to_csv(raw_data_dir / f"{base_name}_frequency_analysis.csv", index=False, encoding='utf-8-sig')
        
        processed_files.append({
            'file_name': base_name,
            'original_file': str(file_path),
            'variable': primary_var,
            'data_type': 'raw',
            'original_length': len(data),
            'processed_length': len(data_downsampled),
            'fault_type': fault_type,
            'rpm': rpm,
            'sampling_rate': target_fs,
            'bearing_type': bearing_type
        })
        
        # 2. 处理去噪数据
        print(f"    🔧 生成去噪数据文件...")
        
        # 去噪处理
        data_denoised = self.apply_denoising(data_downsampled, target_fs)
        
        # 提取去噪数据特征
        denoised_time_features = self.extract_time_features(data_denoised)
        denoised_freq_features, denoised_fft_magnitude, denoised_frequencies = self.compute_fft_features(data_denoised, target_fs)
        denoised_freq_analysis = self.analyze_frequency_components(denoised_fft_magnitude, denoised_frequencies, target_fs, rpm, bearing_type)
        
        # 创建去噪数据文件夹
        denoised_name = f"{base_name}_denoised"
        denoised_data_dir = self.output_base / denoised_name
        denoised_data_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存去噪数据
        np.save(denoised_data_dir / f"{denoised_name}_raw_data.npy", data_denoised)
        
        # 绘制去噪数据图像
        self.plot_time_domain(data_denoised, target_fs, 
                            denoised_data_dir / f"{denoised_name}_time_domain.png", 
                            base_name, "去噪数据")
        self.plot_frequency_domain(denoised_fft_magnitude, denoised_frequencies, 
                                 denoised_data_dir / f"{denoised_name}_frequency_domain.png", 
                                 base_name, "去噪数据", denoised_freq_analysis)
        
        # 保存去噪数据特征
        denoised_all_features = {**denoised_time_features, **denoised_freq_features, 'fault_type': fault_type, 'data_type': 'denoised'}
        denoised_features_df = pd.DataFrame([denoised_all_features])
        denoised_features_df.to_csv(denoised_data_dir / f"{denoised_name}_features.csv", index=False, encoding='utf-8-sig')
        
        # 保存去噪数据频率分析
        denoised_freq_analysis['data_type'] = 'denoised'
        denoised_freq_df = pd.DataFrame([denoised_freq_analysis])
        denoised_freq_df.to_csv(denoised_data_dir / f"{denoised_name}_frequency_analysis.csv", index=False, encoding='utf-8-sig')
        
        processed_files.append({
            'file_name': denoised_name,
            'original_file': str(file_path),
            'variable': primary_var,
            'data_type': 'denoised',
            'original_length': len(data),
            'processed_length': len(data_denoised),
            'fault_type': fault_type,
            'rpm': rpm,
            'sampling_rate': target_fs,
            'bearing_type': bearing_type
        })
        
        self.processed_count += 2  # 每个原始文件生成2个处理后文件
        print(f"    ✅ 完成: {base_name} (原始) + {denoised_name} (去噪)")
        
        return processed_files
    
    def _determine_fault_type(self, filename, category):
        """确定故障类型"""
        filename_upper = filename.upper()
        if 'OR' in filename_upper or 'OUTER' in filename_upper:
            return 'OR'
        elif 'IR' in filename_upper or 'INNER' in filename_upper:
            return 'IR'
        elif 'B' in filename_upper and any(x in filename for x in ['007', '014', '021', '028']):
            return 'B'
        elif 'N' in filename_upper or 'NORMAL' in filename_upper:
            return 'N'
        else:
            return 'Unknown'
    
    # 辅助函数
    def _skewness(self, data):
        """计算偏度"""
        return np.mean(((data - np.mean(data)) / np.std(data)) ** 3)
    
    def _kurtosis(self, data):
        """计算峭度"""
        return np.mean(((data - np.mean(data)) / np.std(data)) ** 4) - 3
    
    def _calculate_entropy(self, data):
        """计算熵"""
        hist, _ = np.histogram(data, bins=50)
        hist = hist[hist > 0]
        prob = hist / np.sum(hist)
        return -np.sum(prob * np.log2(prob))
    
    def _mean_frequency(self, data):
        """计算平均频率"""
        fft_data = np.fft.fft(data)
        magnitude = np.abs(fft_data[:len(fft_data)//2])
        freqs = np.arange(len(magnitude))
        return np.sum(freqs * magnitude) / np.sum(magnitude)
    
    def _freq_skewness(self, frequencies, magnitude):
        """频域偏度"""
        mean_freq = np.sum(frequencies * magnitude) / np.sum(magnitude)
        variance = np.sum((frequencies - mean_freq)**2 * magnitude) / np.sum(magnitude)
        if variance == 0:
            return 0
        std_freq = np.sqrt(variance)
        return np.sum(((frequencies - mean_freq) / std_freq)**3 * magnitude) / np.sum(magnitude)
    
    def _freq_kurtosis(self, frequencies, magnitude):
        """频域峭度"""
        mean_freq = np.sum(frequencies * magnitude) / np.sum(magnitude)
        variance = np.sum((frequencies - mean_freq)**2 * magnitude) / np.sum(magnitude)
        if variance == 0:
            return 0
        std_freq = np.sqrt(variance)
        return np.sum(((frequencies - mean_freq) / std_freq)**4 * magnitude) / np.sum(magnitude) - 3
    
    def _spectral_rolloff(self, frequencies, magnitude, rolloff_percent=0.85):
        """谱滚降"""
        total_energy = np.sum(magnitude)
        rolloff_energy = rolloff_percent * total_energy
        cumulative_energy = np.cumsum(magnitude)
        rolloff_idx = np.where(cumulative_energy >= rolloff_energy)[0]
        return frequencies[rolloff_idx[0]] if len(rolloff_idx) > 0 else frequencies[-1]
    
    def _spectral_flatness(self, magnitude):
        """谱平坦度"""
        geometric_mean = np.exp(np.mean(np.log(magnitude + 1e-10)))
        arithmetic_mean = np.mean(magnitude)
        return geometric_mean / arithmetic_mean if arithmetic_mean > 0 else 0
    
    def _spectral_bandwidth(self, frequencies, magnitude):
        """谱带宽"""
        centroid = np.sum(frequencies * magnitude) / np.sum(magnitude)
        return np.sqrt(np.sum((frequencies - centroid)**2 * magnitude) / np.sum(magnitude))


def main():
    """主函数"""
    print("🔧 原始数据处理器")
    print("=" * 60)
    print("目标: 161个原始文件 → 322个完整数据文件")
    print("方案: 每个文件生成原始版本 + 去噪版本")
    print("=" * 60)
    
    # 路径设置
    source_base = "/Users/gsyoung/MBP_documents/code/Matlab/研究生数学建模/数据集/数据集/源域数据集"
    output_base = "/Users/gsyoung/MBP_documents/code/Matlab/研究生数学建模/建模实现/处理后数据/raw_data"
    
    # 创建处理器
    processor = RawDataProcessor(source_base, output_base)
    
    # 创建输出目录
    Path(output_base).mkdir(parents=True, exist_ok=True)
    
    # 删除旧的处理结果（如果存在）
    if Path(output_base).exists():
        import shutil
        for item in Path(output_base).iterdir():
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()
    
    all_processed_files = []
    processing_stats = {}
    total_input_files = 0
    
    # 处理所有类别
    categories = ['12kHz_DE_data', '12kHz_FE_data', '48kHz_DE_data', '48kHz_Normal_data']
    
    for category in categories:
        category_path = Path(source_base) / category
        if not category_path.exists():
            print(f"⚠️ 类别路径不存在: {category_path}")
            continue
        
        print(f"\n📁 处理类别: {category}")
        
        # 查找所有.mat文件
        mat_files = list(category_path.rglob("*.mat"))
        print(f"找到 {len(mat_files)} 个.mat文件")
        
        category_files = []
        
        for file_idx, mat_file in enumerate(mat_files):
            files = processor.process_single_file(mat_file, category, file_idx)
            category_files.extend(files)
            all_processed_files.extend(files)
            total_input_files += 1
        
        processing_stats[category] = {
            'input_files': len(mat_files),
            'output_files': len(category_files),
            'files_per_input': len(category_files) / len(mat_files) if len(mat_files) > 0 else 0
        }
        
        print(f"📊 {category} 统计:")
        print(f"  输入文件: {len(mat_files)} 个")
        print(f"  输出文件: {len(category_files)} 个")
        print(f"  转换比例: 1:{len(category_files) / len(mat_files):.1f}")
    
    # 保存处理统计
    stats_summary = {
        'processing_time': datetime.now().isoformat(),
        'total_input_files': total_input_files,
        'total_output_files': len(all_processed_files),
        'target_output_files': 322,  # 161 * 2
        'category_stats': processing_stats,
        'files_detail': all_processed_files
    }
    
    # 保存到文件
    reports_dir = Path(output_base).parent / "reports"
    reports_dir.mkdir(exist_ok=True)
    
    with open(reports_dir / "raw_data_processing_report.json", 'w', encoding='utf-8') as f:
        json.dump(stats_summary, f, ensure_ascii=False, indent=2, default=str)
    
    # 保存详细统计CSV
    files_df = pd.DataFrame(all_processed_files)
    files_df.to_csv(reports_dir / "raw_data_summary.csv", index=False, encoding='utf-8-sig')
    
    print(f"\n✅ 处理完成!")
    print(f"📊 总结:")
    print(f"  输入文件: {total_input_files} 个")
    print(f"  输出文件: {len(all_processed_files)} 个")
    print(f"  目标文件: 322 个 (161×2)")
    print(f"  完成率: {len(all_processed_files)/322*100:.1f}%")
    print(f"📁 输出目录: {output_base}")
    print(f"📄 处理报告: raw_data_processing_report.json")


if __name__ == "__main__":
    main()
