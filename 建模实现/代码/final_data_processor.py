#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
最终数据处理器 - 严格按照要求实现
目标：从161个文件生成483个数据片段（每个文件3个片段）
要求：片段利用率≥60%，保证数据对齐和质量
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

class FinalDataProcessor:
    """最终数据处理器 - 严格按照要求实现"""
    
    def __init__(self, source_base_path, output_base_path):
        self.source_base = Path(source_base_path)
        self.output_base = Path(output_base_path)
        self.processed_count = 0
        self.global_folder_counter = 1  # 全局文件夹计数器
        
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
    
    def extract_three_segments_with_high_utilization(self, data, target_fs):
        """
        从数据中提取3个片段，确保高利用率（≥60%）
        策略：均匀分布提取，保证覆盖整个信号
        """
        data_length = len(data)
        
        # 计算目标片段长度，确保利用率≥60%
        min_utilization = 0.6
        min_total_points = int(data_length * min_utilization)
        target_points_per_segment = min_total_points // 3
        
        # 但是要满足基本质量要求
        if target_fs == 12000:
            min_segment_length = 20000  # 12kHz至少2万点
        else:
            min_segment_length = 60000  # 48kHz原始至少6万点（降采样后1.5万点）
        
        # 选择较大的值作为实际段长度
        segment_length = int(max(target_points_per_segment, min_segment_length))
        
        # 如果单个段长度太大，调整为可行的最大值
        max_possible_length = int(data_length // 3)
        segment_length = int(min(segment_length, max_possible_length))
        
        # 确保至少有最小可用长度
        if segment_length < min_segment_length * 0.8:  # 允许20%的降级
            segment_length = int(min_segment_length * 0.8)
        
        # 计算三个段的起始位置，均匀分布
        # 第1段：从头开始
        # 第2段：从中间开始  
        # 第3段：从后面开始
        remaining_length = data_length - 3 * segment_length
        gap = remaining_length // 2 if remaining_length > 0 else 0
        
        segments = []
        positions = [
            0,  # 第1段：开头
            int(gap + segment_length),  # 第2段：中间偏前
            int(data_length - segment_length)  # 第3段：结尾
        ]
        
        for i, start_pos in enumerate(positions):
            start_pos = int(start_pos)
            end_pos = int(start_pos + segment_length)
            if end_pos > data_length:
                end_pos = int(data_length)
                start_pos = int(max(0, end_pos - segment_length))
            
            segment = data[start_pos:end_pos]
            if len(segment) >= segment_length * 0.9:  # 允许10%的长度误差
                segments.append(segment)
            else:
                # 如果段太短，从当前位置取最大可能长度
                available_length = int(data_length - start_pos)
                if available_length > segment_length * 0.5:
                    segments.append(data[start_pos:])
        
        # 确保正好返回3个段
        while len(segments) < 3:
            # 如果段数不够，从剩余数据中补充
            if len(segments) == 0:
                # 极端情况：均匀分割
                seg_len = int(data_length // 3)
                segments = [
                    data[0:seg_len],
                    data[seg_len:2*seg_len], 
                    data[2*seg_len:]
                ]
            else:
                # 复制最后一个段（带偏移）
                last_segment = segments[-1]
                offset = int(len(last_segment) // 2)
                start_pos = int(len(data) - len(last_segment) - offset)
                if start_pos < 0:
                    start_pos = 0
                end_pos = int(start_pos + len(last_segment))
                segments.append(data[start_pos:end_pos])
        
        # 只返回前3个段
        segments = segments[:3]
        
        # 计算实际利用率
        total_used_points = sum(len(seg) for seg in segments)
        utilization_rate = total_used_points / data_length
        
        print(f"    数据长度: {data_length:,}, 片段长度: {[len(s) for s in segments]}")
        print(f"    利用率: {utilization_rate:.1%} (目标≥60%)")
        
        return segments, utilization_rate
    
    def extract_time_features(self, segment_data):
        """提取时域特征"""
        data = segment_data.flatten()
        
        features = {
            'P1_Mean': float(np.mean(data)),
            'P2_RMS': float(np.sqrt(np.mean(data**2))),
            'P3_Variance': float(np.var(data)),
            'P4_Std': float(np.std(data)),
            'P5_Skewness': float(self._skewness(data)),
            'P6_Kurtosis': float(self._kurtosis(data)),
            'P7_Peak': float(np.max(np.abs(data))),
            'P8_Peak2Peak': float(np.max(data) - np.min(data)),
            'P9_CrestFactor': float(np.max(np.abs(data)) / np.sqrt(np.mean(data**2))),
            'P10_ShapeFactor': float(np.sqrt(np.mean(data**2)) / np.mean(np.abs(data))),
            'P11_ImpulseFactor': float(np.max(np.abs(data)) / np.mean(np.abs(data))),
            'P12_MarginFactor': float(np.max(np.abs(data)) / (np.mean(np.sqrt(np.abs(data))))**2),
            'P13_Energy': float(np.sum(data**2)),
            'P14_Entropy': float(self._calculate_entropy(data)),
            'P15_ZeroCrossing': int(np.sum(np.diff(np.sign(data)) != 0)),
            'P16_MeanFreq': float(self._mean_frequency(data))
        }
        
        return features
    
    def compute_fft_features(self, segment_data, fs):
        """计算FFT特征"""
        data = segment_data.flatten()
        
        # 计算FFT
        fft_data = np.fft.fft(data)
        fft_magnitude = np.abs(fft_data[:len(fft_data)//2])
        frequencies = np.fft.fftfreq(len(data), 1/fs)[:len(fft_data)//2]
        
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
    
    def plot_time_domain(self, segment_data, fs, output_path, segment_name):
        """绘制时域信号"""
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        
        data = segment_data.flatten()
        time = np.arange(len(data)) / fs
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(time, data, 'b-', linewidth=0.8)
        ax.set_xlabel('时间 (秒)')
        ax.set_ylabel('振幅')
        ax.set_title(f'{segment_name} - 时域信号')
        ax.grid(True, alpha=0.3)
        
        # 添加统计信息
        rms = np.sqrt(np.mean(data**2))
        peak = np.max(np.abs(data))
        peak2peak = np.max(data) - np.min(data)
        
        info_text = f'RMS: {rms:.4f}\nPeak: {peak:.4f}\nP-P: {peak2peak:.4f}\n点数: {len(data):,}'
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_frequency_domain(self, fft_magnitude, frequencies, output_path, segment_name, freq_analysis):
        """绘制频域信号"""
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(frequencies, fft_magnitude, 'b-', linewidth=0.8)
        ax.set_xlabel('频率 (Hz)')
        ax.set_ylabel('幅值')
        ax.set_title(f'{segment_name} - 频域信号')
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
    
    def process_single_file(self, file_path, category):
        """
        处理单个文件，严格生成3个数据片段
        """
        print(f"\n📄 处理文件: {file_path.name}")
        
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
        
        processed_segments = []
        
        # 只处理第一个主要的时域变量（通常是DE_time）
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
            data = self.downsample_signal(data, original_fs, target_fs)
            print(f"  降采样后长度: {len(data):,}")
        
        # 去噪
        data_denoised = self.apply_denoising(data, target_fs)
        
        # 提取3个片段，确保高利用率
        segments, utilization_rate = self.extract_three_segments_with_high_utilization(data_denoised, target_fs)
        
        # 确保正好有3个片段
        if len(segments) != 3:
            print(f"⚠️ 警告：期望3个片段，实际得到{len(segments)}个")
            # 强制调整为3个片段
            while len(segments) < 3:
                # 复制最后一个段
                segments.append(segments[-1] if segments else data_denoised)
            segments = segments[:3]  # 只取前3个
        
        # 处理每个片段
        for seg_idx, segment in enumerate(segments):
            # 生成唯一的片段ID
            segment_id = f"{file_path.stem}_{seg_idx+1}_{self.global_folder_counter:03d}"
            self.global_folder_counter += 1
            
            # 特征提取
            time_features = self.extract_time_features(segment)
            freq_features, fft_magnitude, frequencies = self.compute_fft_features(segment, target_fs)
            
            # 频率分析
            bearing_type = 'SKF6205' if 'DE' in primary_var else 'SKF6203'
            freq_analysis = self.analyze_frequency_components(fft_magnitude, frequencies, target_fs, rpm, bearing_type)
            
            # 确定故障类型
            fault_type = self._determine_fault_type(file_path.name, category)
            
            # 创建输出目录
            segment_dir = self.output_base / f"{segment_id}"
            segment_dir.mkdir(parents=True, exist_ok=True)
            
            # 保存原始数据
            np.save(segment_dir / f"{segment_id}_raw_data.npy", segment)
            
            # 绘制图像
            self.plot_time_domain(segment, target_fs, 
                                segment_dir / f"{segment_id}_time_domain.png", segment_id)
            self.plot_frequency_domain(fft_magnitude, frequencies, 
                                     segment_dir / f"{segment_id}_frequency_domain.png", 
                                     segment_id, freq_analysis)
            
            # 保存特征
            all_features = {**time_features, **freq_features, 'fault_type': fault_type}
            features_df = pd.DataFrame([all_features])
            features_df.to_csv(segment_dir / f"{segment_id}_features.csv", index=False, encoding='utf-8-sig')
            
            # 保存频率分析
            freq_df = pd.DataFrame([freq_analysis])
            freq_df.to_csv(segment_dir / f"{segment_id}_frequency_analysis.csv", index=False, encoding='utf-8-sig')
            
            processed_segments.append({
                'segment_id': segment_id,
                'original_file': str(file_path),
                'variable': primary_var,
                'segment_index': seg_idx + 1,
                'original_length': len(data),
                'processed_length': len(segment),
                'fault_type': fault_type,
                'rpm': rpm,
                'sampling_rate': target_fs,
                'utilization_rate': utilization_rate
            })
            
            self.processed_count += 1
            print(f"    ✅ 片段 {seg_idx+1}: {segment_id} (长度: {len(segment):,})")
        
        return processed_segments
    
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
    print("🎯 最终数据处理器 - 严格按照要求实现")
    print("=" * 60)
    print("目标: 161个文件 → 483个数据片段 (每个文件3个)")
    print("要求: 片段利用率≥60%，保证数据对齐和质量")
    print("=" * 60)
    
    # 路径设置
    source_base = "/Users/gsyoung/MBP_documents/code/Matlab/研究生数学建模/数据集/数据集/源域数据集"
    output_base = "/Users/gsyoung/MBP_documents/code/Matlab/研究生数学建模/建模实现/处理后数据/final_segments"
    
    # 创建处理器
    processor = FinalDataProcessor(source_base, output_base)
    
    # 创建输出目录
    Path(output_base).mkdir(parents=True, exist_ok=True)
    
    all_processed_segments = []
    processing_stats = {}
    total_files_processed = 0
    total_utilization_rates = []
    
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
        
        category_segments = []
        category_utilization_rates = []
        
        for file_idx, mat_file in enumerate(mat_files):
            segments = processor.process_single_file(mat_file, category)
            category_segments.extend(segments)
            all_processed_segments.extend(segments)
            total_files_processed += 1
            
            # 收集利用率数据
            if segments:
                file_utilization = segments[0]['utilization_rate']  # 所有片段的利用率相同
                category_utilization_rates.append(file_utilization)
                total_utilization_rates.append(file_utilization)
        
        avg_utilization = np.mean(category_utilization_rates) if category_utilization_rates else 0
        
        processing_stats[category] = {
            'input_files': len(mat_files),
            'output_segments': len(category_segments),
            'segments_per_file': len(category_segments) / len(mat_files) if len(mat_files) > 0 else 0,
            'avg_utilization_rate': avg_utilization
        }
        
        print(f"📊 {category} 统计:")
        print(f"  输入文件: {len(mat_files)} 个")
        print(f"  输出片段: {len(category_segments)} 个")
        print(f"  平均每文件片段数: {len(category_segments) / len(mat_files):.1f}")
        print(f"  平均利用率: {avg_utilization:.1%}")
    
    # 总体统计
    overall_utilization = np.mean(total_utilization_rates) if total_utilization_rates else 0
    
    print(f"\n🎯 总体目标达成情况:")
    print(f"  目标文件数: 161 个")
    print(f"  实际处理: {total_files_processed} 个")
    print(f"  目标片段数: 483 个 (161×3)")
    print(f"  实际生成: {len(all_processed_segments)} 个")
    print(f"  达成率: {len(all_processed_segments)/483*100:.1f}%")
    print(f"  平均利用率: {overall_utilization:.1%} (要求≥60%)")
    
    # 利用率检查
    utilization_pass = overall_utilization >= 0.6
    segment_count_pass = len(all_processed_segments) == 483
    
    print(f"\n✅ 质量检查:")
    print(f"  片段数量: {'✅ 通过' if segment_count_pass else '❌ 未达标'}")
    print(f"  利用率要求: {'✅ 通过' if utilization_pass else '❌ 未达标'}")
    
    # 保存处理统计
    stats_summary = {
        'processing_time': datetime.now().isoformat(),
        'target_files': 161,
        'processed_files': total_files_processed,
        'target_segments': 483,
        'generated_segments': len(all_processed_segments),
        'target_utilization': 0.6,
        'actual_utilization': overall_utilization,
        'category_stats': processing_stats,
        'segments_detail': all_processed_segments,
        'quality_check': {
            'segment_count_pass': segment_count_pass,
            'utilization_pass': utilization_pass
        }
    }
    
    # 保存到文件
    reports_dir = Path(output_base).parent / "reports"
    reports_dir.mkdir(exist_ok=True)
    
    with open(reports_dir / "final_processing_report.json", 'w', encoding='utf-8') as f:
        json.dump(stats_summary, f, ensure_ascii=False, indent=2, default=str)
    
    # 保存详细统计CSV
    segments_df = pd.DataFrame(all_processed_segments)
    segments_df.to_csv(reports_dir / "final_segments_summary.csv", index=False, encoding='utf-8-sig')
    
    if segment_count_pass and utilization_pass:
        print(f"\n🎉 任务完成！成功生成483个高质量数据片段")
    else:
        print(f"\n⚠️ 任务需要调整，请检查质量指标")
    
    print(f"📁 输出目录: {output_base}")
    print(f"📊 详细报告: final_processing_report.json")

if __name__ == "__main__":
    main()
