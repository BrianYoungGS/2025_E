#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强版源域数据集处理程序
- 解决文件夹名称冲突问题
- 增加主频检测和标注功能
- 增加谐波成分分析
- 生成频率分析CSV文件
- 确保生成322个唯一的数据文件夹
"""

import os
import numpy as np
import scipy.io as sio
import scipy.signal as signal
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

class EnhancedBearingDataProcessor:
    """增强版轴承数据处理器"""
    
    def __init__(self, base_path, output_path):
        self.base_path = Path(base_path)
        self.output_path = Path(output_path)
        self.source_path = self.base_path / "数据集" / "数据集" / "源域数据集"
        
        # 创建输出目录
        self.output_path.mkdir(parents=True, exist_ok=True)
        (self.output_path / "processed_segments").mkdir(exist_ok=True)
        (self.output_path / "reports").mkdir(exist_ok=True)
        
        # 处理参数
        self.target_fs = 12000  # 目标采样频率
        self.segment_length = 2048  # 数据片段长度
        
        # 滤波参数
        self.filter_params = {
            'highpass_cutoff': 10,    # 高通滤波截止频率(Hz)
            'lowpass_cutoff': 5000,   # 低通滤波截止频率(Hz)
            'notch_freq': 50,         # 陷波滤波频率(Hz)
            'notch_q': 30             # 陷波滤波品质因数
        }
        
        # 轴承参数（用于理论频率计算）
        self.bearing_params = {
            'SKF6205': {  # 驱动端轴承
                'balls': 9,
                'ball_diameter': 0.3126,  # 英寸
                'pitch_diameter': 1.537,  # 英寸
            },
            'SKF6203': {  # 风扇端轴承
                'balls': 9,
                'ball_diameter': 0.2656,  # 英寸
                'pitch_diameter': 1.122,  # 英寸
            }
        }
        
        # 特征计算参数
        self.feature_names = [
            'P1_Mean', 'P2_RMS', 'P3_SquareRootAmplitude', 'P4_AbsoluteAverage',
            'P5_Maximum', 'P6_Minimum', 'P7_PeakToPeak', 'P8_Variance',
            'P9_Skewness', 'P10_Kurtosis', 'P11_ShapeFactor', 'P12_CrestFactor',
            'P13_ImpulseFactor', 'P14_CoefficientOfVariation', 'P15_CoefficientOfSkewness',
            'P16_CoefficientOfKurtosis', 'P17_MeanFrequency', 'P18_SpectralVariance',
            'P19_SpectralSkewness', 'P20_SpectralKurtosis', 'P21_FrequencyCenter',
            'P22_StandardDeviationFrequency', 'P23_RootMeanSquareFrequency',
            'P24_P25_MainFrequencyBandPosition1', 'P25_MainFrequencyBandPosition2',
            'P26_SpectrumDispersion1', 'P27_SpectrumDispersion2', 'P28_SpectrumConcentration',
            'P29_SpectrumConcentration2'
        ]
        
        self.processed_files = []
        self.global_segment_counter = 0  # 全局计数器确保唯一性
        
    def generate_unique_segment_name(self, file_path, segment_type, data_category):
        """
        生成唯一的数据片段名称
        
        Args:
            file_path: 文件路径
            segment_type: 片段类型 ('front' 或 'back')
            data_category: 数据类别 ('12kHz_DE', '12kHz_FE', '48kHz_DE', '48kHz_Normal')
            
        Returns:
            unique_name: 唯一的片段名称
        """
        self.global_segment_counter += 1
        base_name = file_path.stem
        unique_name = f"{data_category}_{base_name}_{segment_type}_{self.segment_length}_{self.global_segment_counter:03d}"
        return unique_name
    
    def apply_denoising_filters(self, data, fs):
        """应用去噪滤波器组合"""
        filtered_data = data.copy()
        
        # 1. 高通滤波
        if self.filter_params['highpass_cutoff'] > 0:
            nyquist = fs / 2
            high_cutoff = self.filter_params['highpass_cutoff'] / nyquist
            if high_cutoff < 1.0:
                b_high, a_high = signal.butter(4, high_cutoff, btype='high')
                filtered_data = signal.filtfilt(b_high, a_high, filtered_data)
        
        # 2. 低通滤波
        if self.filter_params['lowpass_cutoff'] < fs/2:
            nyquist = fs / 2
            low_cutoff = self.filter_params['lowpass_cutoff'] / nyquist
            if low_cutoff < 1.0:
                b_low, a_low = signal.butter(4, low_cutoff, btype='low')
                filtered_data = signal.filtfilt(b_low, a_low, filtered_data)
        
        # 3. 陷波滤波
        for harmonic in [1, 2, 3]:
            notch_freq = self.filter_params['notch_freq'] * harmonic
            if notch_freq < fs/2:
                nyquist = fs / 2
                notch_norm = notch_freq / nyquist
                if 0 < notch_norm < 1:
                    b_notch, a_notch = signal.iirnotch(notch_norm, self.filter_params['notch_q'])
                    filtered_data = signal.filtfilt(b_notch, a_notch, filtered_data)
        
        # 4. 中值滤波
        if len(filtered_data) > 5:
            filtered_data = signal.medfilt(filtered_data, kernel_size=3)
        
        return filtered_data
    
    def downsample_data(self, data, original_fs, target_fs):
        """降采样数据"""
        if original_fs == target_fs:
            return data
            
        downsample_ratio = original_fs / target_fs
        new_length = int(len(data) / downsample_ratio)
        downsampled_data = signal.resample(data, new_length)
        
        return downsampled_data
    
    def extract_segments(self, data, segment_length):
        """从数据中提取前后两个片段"""
        data_length = len(data)
        
        if data_length < segment_length * 2:
            padded_data = np.pad(data, (0, segment_length * 2 - data_length), 'constant')
            front_segment = padded_data[:segment_length]
            back_segment = padded_data[-segment_length:]
        else:
            front_segment = data[:segment_length]
            back_segment = data[-segment_length:]
            
        return front_segment, back_segment
    
    def detect_dominant_frequencies(self, magnitude, frequencies, num_peaks=5):
        """
        检测主频和谐波成分
        
        Args:
            magnitude: 频谱幅值
            frequencies: 频率数组
            num_peaks: 检测的峰值数量
            
        Returns:
            peak_info: 包含主频和谐波信息的字典
        """
        # 使用find_peaks检测峰值
        peaks, properties = find_peaks(magnitude, 
                                     height=np.max(magnitude) * 0.1,  # 至少是最大值的10%
                                     distance=10,  # 峰值间距
                                     prominence=np.max(magnitude) * 0.05)  # 突出度
        
        if len(peaks) == 0:
            return {
                'dominant_frequency': 0,
                'dominant_amplitude': 0,
                'harmonics': [],
                'peak_frequencies': [],
                'peak_amplitudes': []
            }
        
        # 按幅值排序，选择最强的几个峰值
        peak_amplitudes = magnitude[peaks]
        sorted_indices = np.argsort(peak_amplitudes)[::-1]
        
        # 选择前num_peaks个最强峰值
        top_peaks = peaks[sorted_indices[:min(num_peaks, len(peaks))]]
        top_amplitudes = magnitude[top_peaks]
        top_frequencies = frequencies[top_peaks]
        
        # 主频（最强峰值）
        dominant_frequency = top_frequencies[0]
        dominant_amplitude = top_amplitudes[0]
        
        # 检测谐波（频率是主频整数倍的峰值）
        harmonics = []
        for i in range(1, len(top_frequencies)):
            freq = top_frequencies[i]
            amp = top_amplitudes[i]
            
            # 检查是否为主频的谐波（允许±5%的误差）
            harmonic_order = round(freq / dominant_frequency)
            if harmonic_order > 1 and abs(freq - harmonic_order * dominant_frequency) / dominant_frequency < 0.05:
                harmonics.append({
                    'order': harmonic_order,
                    'frequency': freq,
                    'amplitude': amp,
                    'amplitude_ratio': amp / dominant_amplitude
                })
        
        return {
            'dominant_frequency': dominant_frequency,
            'dominant_amplitude': dominant_amplitude,
            'harmonics': harmonics,
            'peak_frequencies': top_frequencies.tolist(),
            'peak_amplitudes': top_amplitudes.tolist()
        }
    
    def calculate_theoretical_fault_frequencies(self, rpm, bearing_type='SKF6205'):
        """
        计算理论故障频率
        
        Args:
            rpm: 转速
            bearing_type: 轴承类型
            
        Returns:
            fault_freqs: 理论故障频率字典
        """
        if rpm is None or rpm <= 0:
            return {}
            
        params = self.bearing_params.get(bearing_type, self.bearing_params['SKF6205'])
        
        # 转换为Hz
        shaft_freq = rpm / 60
        
        # 接触角假设为0度（径向载荷）
        contact_angle = 0
        cos_alpha = np.cos(np.radians(contact_angle))
        
        # 计算故障频率
        # BPFO: Ball Pass Frequency Outer race
        bpfo = (params['balls'] / 2) * shaft_freq * (1 - (params['ball_diameter'] / params['pitch_diameter']) * cos_alpha)
        
        # BPFI: Ball Pass Frequency Inner race  
        bpfi = (params['balls'] / 2) * shaft_freq * (1 + (params['ball_diameter'] / params['pitch_diameter']) * cos_alpha)
        
        # BSF: Ball Spin Frequency
        bsf = (params['pitch_diameter'] / (2 * params['ball_diameter'])) * shaft_freq * (1 - (params['ball_diameter'] / params['pitch_diameter'])**2 * cos_alpha**2)
        
        # FTF: Fundamental Train Frequency (cage frequency)
        ftf = shaft_freq / 2 * (1 - (params['ball_diameter'] / params['pitch_diameter']) * cos_alpha)
        
        return {
            'shaft_frequency': shaft_freq,
            'BPFO': bpfo,  # 外圈故障频率
            'BPFI': bpfi,  # 内圈故障频率
            'BSF': bsf,    # 滚动体故障频率
            'FTF': ftf     # 保持架频率
        }
    
    def calculate_time_domain_features(self, signal_data):
        """计算时域特征 (P1-P16)"""
        features = {}
        
        # 基本统计量
        features['P1_Mean'] = np.mean(signal_data)
        features['P2_RMS'] = np.sqrt(np.mean(signal_data**2))
        features['P3_SquareRootAmplitude'] = (np.mean(np.sqrt(np.abs(signal_data))))**2
        features['P4_AbsoluteAverage'] = np.mean(np.abs(signal_data))
        features['P5_Maximum'] = np.max(signal_data)
        features['P6_Minimum'] = np.min(signal_data)
        features['P7_PeakToPeak'] = features['P5_Maximum'] - features['P6_Minimum']
        features['P8_Variance'] = np.var(signal_data)
        
        # 高阶统计量
        if features['P8_Variance'] > 0:
            features['P9_Skewness'] = np.mean((signal_data - features['P1_Mean'])**3) / (features['P8_Variance']**1.5)
            features['P10_Kurtosis'] = np.mean((signal_data - features['P1_Mean'])**4) / (features['P8_Variance']**2)
        else:
            features['P9_Skewness'] = 0
            features['P10_Kurtosis'] = 0
        
        # 形状因子
        if features['P4_AbsoluteAverage'] > 0:
            features['P11_ShapeFactor'] = features['P2_RMS'] / features['P4_AbsoluteAverage']
        else:
            features['P11_ShapeFactor'] = 0
            
        if features['P2_RMS'] > 0:
            features['P12_CrestFactor'] = features['P5_Maximum'] / features['P2_RMS']
        else:
            features['P12_CrestFactor'] = 0
            
        if features['P4_AbsoluteAverage'] > 0:
            features['P13_ImpulseFactor'] = features['P5_Maximum'] / features['P4_AbsoluteAverage']
        else:
            features['P13_ImpulseFactor'] = 0
        
        # 变异系数
        if features['P1_Mean'] != 0:
            features['P14_CoefficientOfVariation'] = np.sqrt(features['P8_Variance']) / abs(features['P1_Mean'])
        else:
            features['P14_CoefficientOfVariation'] = 0
            
        if features['P8_Variance'] > 0:
            features['P15_CoefficientOfSkewness'] = features['P9_Skewness'] / (features['P8_Variance']**0.5)
            features['P16_CoefficientOfKurtosis'] = features['P10_Kurtosis'] / (features['P8_Variance']**2)
        else:
            features['P15_CoefficientOfSkewness'] = 0
            features['P16_CoefficientOfKurtosis'] = 0
            
        return features
    
    def calculate_frequency_domain_features(self, signal_data, fs):
        """计算频域特征 (P17-P29)"""
        features = {}
        
        # FFT变换
        fft_data = np.fft.fft(signal_data)
        magnitude = np.abs(fft_data[:len(fft_data)//2])
        frequencies = np.fft.fftfreq(len(signal_data), 1/fs)[:len(fft_data)//2]
        
        # 功率谱密度
        psd = magnitude**2
        
        # 归一化功率谱
        total_power = np.sum(psd)
        if total_power > 0:
            normalized_psd = psd / total_power
        else:
            normalized_psd = psd
            
        # P17: 平均频率
        if total_power > 0:
            features['P17_MeanFrequency'] = np.sum(frequencies * normalized_psd)
        else:
            features['P17_MeanFrequency'] = 0
        
        # P18: 频谱方差
        mean_freq = features['P17_MeanFrequency']
        features['P18_SpectralVariance'] = np.sum(((frequencies - mean_freq)**2) * normalized_psd)
        
        # P19: 频谱偏度
        if features['P18_SpectralVariance'] > 0:
            features['P19_SpectralSkewness'] = np.sum(((frequencies - mean_freq)**3) * normalized_psd) / (features['P18_SpectralVariance']**1.5)
        else:
            features['P19_SpectralSkewness'] = 0
        
        # P20: 频谱峰度
        if features['P18_SpectralVariance'] > 0:
            features['P20_SpectralKurtosis'] = np.sum(((frequencies - mean_freq)**4) * normalized_psd) / (features['P18_SpectralVariance']**2)
        else:
            features['P20_SpectralKurtosis'] = 0
        
        # P21: 频率重心
        features['P21_FrequencyCenter'] = features['P17_MeanFrequency']
        
        # P22: 频率标准差
        features['P22_StandardDeviationFrequency'] = np.sqrt(features['P18_SpectralVariance'])
        
        # P23: 频率均方根
        features['P23_RootMeanSquareFrequency'] = np.sqrt(np.sum((frequencies**2) * normalized_psd))
        
        # P24-P25: 主频带位置指示器
        max_power_idx = np.argmax(psd)
        features['P24_P25_MainFrequencyBandPosition1'] = frequencies[max_power_idx] if len(frequencies) > max_power_idx else 0
        
        if len(psd) > 1:
            second_max_idx = np.argsort(psd)[-2]
            features['P25_MainFrequencyBandPosition2'] = frequencies[second_max_idx] if len(frequencies) > second_max_idx else 0
        else:
            features['P25_MainFrequencyBandPosition2'] = 0
        
        # P26-P29: 频谱分散度和集中度指示器
        if total_power > 0:
            features['P26_SpectrumDispersion1'] = np.sum(((frequencies - features['P21_FrequencyCenter'])**2) * psd) / total_power
            features['P27_SpectrumDispersion2'] = np.sum(((frequencies - features['P21_FrequencyCenter'])**4) * psd) / total_power
            features['P28_SpectrumConcentration'] = np.sum(((frequencies - features['P21_FrequencyCenter'])**4) * psd) / (features['P26_SpectrumDispersion1']**2) if features['P26_SpectrumDispersion1'] > 0 else 0
            features['P29_SpectrumConcentration2'] = np.sum(np.sqrt((frequencies - features['P21_FrequencyCenter'])**2) * psd) / (total_power * np.sqrt(features['P22_StandardDeviationFrequency'])) if features['P22_StandardDeviationFrequency'] > 0 else 0
        else:
            features['P26_SpectrumDispersion1'] = 0
            features['P27_SpectrumDispersion2'] = 0
            features['P28_SpectrumConcentration'] = 0
            features['P29_SpectrumConcentration2'] = 0
            
        return features, magnitude, frequencies
    
    def determine_fault_type(self, file_path):
        """根据文件路径确定故障类型"""
        path_str = str(file_path)
        
        if '/B/' in path_str or 'B0' in path_str:
            return 'B'  # 滚动体故障
        elif '/IR/' in path_str or 'IR0' in path_str:
            return 'IR'  # 内圈故障
        elif '/OR/' in path_str or 'OR0' in path_str:
            return 'OR'  # 外圈故障
        elif '/N_' in path_str or 'Normal' in path_str:
            return 'N'  # 正常
        else:
            return 'Unknown'
    
    def get_rpm_from_data(self, data, file_path):
        """从数据中获取转速信息"""
        keys = [k for k in data.keys() if not k.startswith('__')]
        
        # 查找RPM变量
        for key in keys:
            if 'RPM' in key.upper():
                rpm_data = data[key]
                if hasattr(rpm_data, '__iter__') and len(rpm_data) > 0:
                    return float(rpm_data.flatten()[0])
        
        # 从文件名中提取转速
        filename = file_path.name
        if '(1797rpm)' in filename:
            return 1797
        elif '(1772rpm)' in filename:
            return 1772
        elif '(1750rpm)' in filename:
            return 1750
        elif '(1730rpm)' in filename:
            return 1730
        
        return None
    
    def plot_time_domain_with_annotations(self, data, title, save_path, peak_info):
        """绘制带标注的时域信号"""
        plt.figure(figsize=(14, 8))
        time_axis = np.arange(len(data)) / self.target_fs
        plt.plot(time_axis, data, 'b-', linewidth=0.8)
        plt.title(f'时域信号 - {title}', fontsize=14, fontweight='bold')
        plt.xlabel('时间 (s)', fontsize=12)
        plt.ylabel('幅值', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # 添加统计信息文本框
        stats_text = f'RMS: {np.sqrt(np.mean(data**2)):.4f}\n'
        stats_text += f'峰值: {np.max(np.abs(data)):.4f}\n'
        stats_text += f'峰峰值: {np.max(data) - np.min(data):.4f}'
        
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def plot_frequency_domain_with_annotations(self, magnitude, frequencies, title, save_path, peak_info, fault_freqs=None):
        """绘制带主频标注的频域信号"""
        plt.figure(figsize=(14, 8))
        plt.semilogy(frequencies, magnitude, 'r-', linewidth=0.8)
        plt.title(f'频域信号 - {title}', fontsize=14, fontweight='bold')
        plt.xlabel('频率 (Hz)', fontsize=12)
        plt.ylabel('幅值', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.xlim(0, max(frequencies))
        
        # 标注主频
        if peak_info['dominant_frequency'] > 0:
            plt.axvline(x=peak_info['dominant_frequency'], color='green', linestyle='--', linewidth=2, alpha=0.8)
            plt.text(peak_info['dominant_frequency'], peak_info['dominant_amplitude'], 
                    f'主频: {peak_info["dominant_frequency"]:.1f}Hz', 
                    rotation=90, verticalalignment='bottom', fontsize=10, fontweight='bold', color='green')
        
        # 标注谐波
        for i, harmonic in enumerate(peak_info['harmonics'][:3]):  # 只标注前3个谐波
            plt.axvline(x=harmonic['frequency'], color='orange', linestyle=':', linewidth=1.5, alpha=0.7)
            plt.text(harmonic['frequency'], harmonic['amplitude'], 
                    f'{harmonic["order"]}次谐波\n{harmonic["frequency"]:.1f}Hz', 
                    rotation=90, verticalalignment='bottom', fontsize=8, color='orange')
        
        # 标注理论故障频率（如果有）
        if fault_freqs and len(fault_freqs) > 0:
            colors = {'BPFO': 'purple', 'BPFI': 'brown', 'BSF': 'pink', 'FTF': 'gray'}
            y_pos = np.max(magnitude) * 0.1
            for freq_type, freq_val in fault_freqs.items():
                if freq_type in colors and freq_val < max(frequencies):
                    plt.axvline(x=freq_val, color=colors[freq_type], linestyle='-.', linewidth=1, alpha=0.6)
                    plt.text(freq_val, y_pos, f'{freq_type}\n{freq_val:.1f}Hz', 
                            rotation=90, verticalalignment='bottom', fontsize=7, color=colors[freq_type])
        
        # 添加图例和信息框
        legend_text = f'主频: {peak_info["dominant_frequency"]:.1f} Hz\n'
        legend_text += f'谐波数量: {len(peak_info["harmonics"])}\n'
        legend_text += f'峰值数量: {len(peak_info["peak_frequencies"])}'
        
        plt.text(0.98, 0.98, legend_text, transform=plt.gca().transAxes, 
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def process_single_file(self, file_path, data_category):
        """处理单个文件"""
        try:
            # 加载数据
            data = sio.loadmat(str(file_path))
            keys = [k for k in data.keys() if not k.startswith('__')]
            
            # 找到主要的信号数据
            de_key = None
            for key in keys:
                if 'DE_time' in key:
                    de_key = key
                    break
            
            if de_key is None:
                print(f"警告: 文件 {file_path.name} 中未找到DE_time数据")
                return []
            
            signal_data = data[de_key].flatten()
            
            # 获取转速信息
            rpm = self.get_rpm_from_data(data, file_path)
            
            # 确定原始采样频率
            if '48kHz' in str(file_path):
                original_fs = 48000
            else:
                original_fs = 12000
            
            # 1. 降采样（如果需要）
            if original_fs != self.target_fs:
                signal_data = self.downsample_data(signal_data, original_fs, self.target_fs)
            
            # 2. 去噪滤波
            filtered_data = self.apply_denoising_filters(signal_data, self.target_fs)
            
            # 3. 提取前后两段数据
            front_segment, back_segment = self.extract_segments(filtered_data, self.segment_length)
            
            # 确定故障类型
            fault_type = self.determine_fault_type(file_path)
            
            # 计算理论故障频率
            bearing_type = 'SKF6205' if 'DE' in data_category else 'SKF6203'
            fault_freqs = self.calculate_theoretical_fault_frequencies(rpm, bearing_type)
            
            # 处理两个数据片段
            segments_info = []
            for i, (segment, suffix) in enumerate([(front_segment, 'front'), (back_segment, 'back')]):
                # 生成唯一的片段名称
                segment_name = self.generate_unique_segment_name(file_path, suffix, data_category)
                segment_dir = self.output_path / "processed_segments" / segment_name
                segment_dir.mkdir(exist_ok=True)
                
                # 4. 特征提取
                time_features = self.calculate_time_domain_features(segment)
                freq_features, magnitude, frequencies = self.calculate_frequency_domain_features(segment, self.target_fs)
                
                # 5. 主频和谐波分析
                peak_info = self.detect_dominant_frequencies(magnitude, frequencies)
                
                # 合并所有特征
                all_features = {**time_features, **freq_features}
                all_features['fault_type'] = fault_type
                all_features['file_name'] = file_path.name
                all_features['segment'] = suffix
                all_features['data_category'] = data_category
                all_features['rpm'] = rpm
                all_features['dominant_frequency'] = peak_info['dominant_frequency']
                all_features['harmonics_count'] = len(peak_info['harmonics'])
                
                # 6. 保存处理后的数据
                np.save(segment_dir / f"{segment_name}_raw_data.npy", segment)
                
                # 7. 绘制时域图
                time_plot_path = segment_dir / f"{segment_name}_time_domain.png"
                self.plot_time_domain_with_annotations(segment, segment_name, time_plot_path, peak_info)
                
                # 8. 绘制频域图
                freq_plot_path = segment_dir / f"{segment_name}_frequency_domain.png"
                self.plot_frequency_domain_with_annotations(magnitude, frequencies, segment_name, 
                                                          freq_plot_path, peak_info, fault_freqs)
                
                # 9. 保存特征CSV
                features_df = pd.DataFrame([all_features])
                features_csv_path = segment_dir / f"{segment_name}_features.csv"
                features_df.to_csv(features_csv_path, index=False, encoding='utf-8-sig')
                
                # 10. 保存频率分析CSV
                freq_analysis = {
                    'segment_name': segment_name,
                    'dominant_frequency': peak_info['dominant_frequency'],
                    'dominant_amplitude': peak_info['dominant_amplitude'],
                    'harmonics_count': len(peak_info['harmonics']),
                    'rpm': rpm,
                    'fault_type': fault_type
                }
                
                # 添加谐波信息
                for j, harmonic in enumerate(peak_info['harmonics'][:5]):  # 最多5个谐波
                    freq_analysis[f'harmonic_{j+1}_order'] = harmonic['order']
                    freq_analysis[f'harmonic_{j+1}_frequency'] = harmonic['frequency']
                    freq_analysis[f'harmonic_{j+1}_amplitude'] = harmonic['amplitude']
                    freq_analysis[f'harmonic_{j+1}_ratio'] = harmonic['amplitude_ratio']
                
                # 添加理论故障频率
                for freq_type, freq_val in fault_freqs.items():
                    freq_analysis[f'theoretical_{freq_type.lower()}'] = freq_val
                
                freq_analysis_df = pd.DataFrame([freq_analysis])
                freq_csv_path = segment_dir / f"{segment_name}_frequency_analysis.csv"
                freq_analysis_df.to_csv(freq_csv_path, index=False, encoding='utf-8-sig')
                
                segments_info.append({
                    'segment_name': segment_name,
                    'segment_dir': str(segment_dir),
                    'fault_type': fault_type,
                    'data_category': data_category,
                    'features': all_features,
                    'frequency_analysis': freq_analysis
                })
                
            return segments_info
            
        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {e}")
            return []
    
    def process_all_files(self):
        """处理所有文件"""
        print("开始处理所有源域数据文件...")
        print(f"去噪滤波参数: {self.filter_params}")
        
        all_segments = []
        total_files = 0
        
        # 遍历所有数据类别
        categories = {
            '12kHz_DE_data': '12kHz_DE',
            '12kHz_FE_data': '12kHz_FE', 
            '48kHz_DE_data': '48kHz_DE',
            '48kHz_Normal_data': '48kHz_Normal'
        }
        
        for category, category_code in categories.items():
            category_path = self.source_path / category
            if not category_path.exists():
                continue
                
            print(f"\n处理类别: {category} -> {category_code}")
            
            # 递归查找所有.mat文件
            mat_files = list(category_path.rglob("*.mat"))
            total_files += len(mat_files)
            
            for file_path in mat_files:
                print(f"  处理文件: {file_path.relative_to(self.source_path)}")
                segments_info = self.process_single_file(file_path, category_code)
                all_segments.extend(segments_info)
        
        self.processed_files = all_segments
        
        # 生成处理报告
        self.generate_processing_report(total_files)
        
        print(f"\n处理完成！")
        print(f"总文件数: {total_files}")
        print(f"生成数据片段数: {len(all_segments)}")
        print(f"输出目录: {self.output_path}")
        
        return all_segments
    
    def generate_processing_report(self, total_files):
        """生成处理报告"""
        report = {
            'processing_time': datetime.now().isoformat(),
            'total_source_files': total_files,
            'total_segments_generated': len(self.processed_files),
            'target_sampling_frequency': self.target_fs,
            'segment_length': self.segment_length,
            'filter_parameters': self.filter_params,
            'feature_count': len(self.feature_names),
            'fault_type_distribution': {},
            'data_category_distribution': {}
        }
        
        # 统计故障类型分布
        fault_types = [seg['fault_type'] for seg in self.processed_files]
        for fault_type in set(fault_types):
            report['fault_type_distribution'][fault_type] = fault_types.count(fault_type)
        
        # 统计数据类别分布
        data_categories = [seg['data_category'] for seg in self.processed_files]
        for category in set(data_categories):
            report['data_category_distribution'][category] = data_categories.count(category)
        
        # 保存JSON报告
        report_path = self.output_path / "reports" / "processing_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        # 生成汇总CSV
        all_features_list = []
        all_freq_analysis_list = []
        
        for seg in self.processed_files:
            features_row = seg['features'].copy()
            features_row['segment_name'] = seg['segment_name']
            all_features_list.append(features_row)
            
            freq_row = seg['frequency_analysis'].copy()
            all_freq_analysis_list.append(freq_row)
        
        if all_features_list:
            summary_df = pd.DataFrame(all_features_list)
            summary_csv_path = self.output_path / "reports" / "all_features_summary.csv"
            summary_df.to_csv(summary_csv_path, index=False, encoding='utf-8-sig')
        
        if all_freq_analysis_list:
            freq_summary_df = pd.DataFrame(all_freq_analysis_list)
            freq_summary_csv_path = self.output_path / "reports" / "all_frequency_analysis_summary.csv"
            freq_summary_df.to_csv(freq_summary_csv_path, index=False, encoding='utf-8-sig')
        
        # 保存文本报告
        text_report_path = self.output_path / "reports" / "processing_report.txt"
        with open(text_report_path, 'w', encoding='utf-8') as f:
            f.write("增强版源域数据处理报告\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"处理时间: {report['processing_time']}\n")
            f.write(f"源文件总数: {report['total_source_files']}\n")
            f.write(f"生成数据片段总数: {report['total_segments_generated']}\n")
            f.write(f"目标采样频率: {report['target_sampling_frequency']} Hz\n")
            f.write(f"数据片段长度: {report['segment_length']} 点\n\n")
            
            f.write("去噪滤波参数:\n")
            for key, value in report['filter_parameters'].items():
                f.write(f"  {key}: {value}\n")
            f.write("\n")
            
            f.write("故障类型分布:\n")
            for fault_type, count in report['fault_type_distribution'].items():
                f.write(f"  {fault_type}: {count} 个片段\n")
            f.write("\n")
            
            f.write("数据类别分布:\n")
            for category, count in report['data_category_distribution'].items():
                f.write(f"  {category}: {count} 个片段\n")
            f.write("\n")
            
            f.write(f"特征维度: {report['feature_count']} 个特征\n")
            f.write("增强功能:\n")
            f.write("  - 主频检测和标注\n")
            f.write("  - 谐波成分分析\n")
            f.write("  - 理论故障频率计算\n")
            f.write("  - 频率分析CSV文件\n")
            f.write("  - 唯一文件夹命名\n")

def main():
    """主函数"""
    # 设置路径
    base_path = "/Users/gsyoung/MBP_documents/code/Matlab/研究生数学建模"
    output_path = "/Users/gsyoung/MBP_documents/code/Matlab/研究生数学建模/建模实现/处理后数据"
    
    print("增强版源域数据处理器")
    print("=" * 50)
    print("新增功能:")
    print("  ✓ 解决文件夹名称冲突")
    print("  ✓ 主频检测和标注")
    print("  ✓ 谐波成分分析")  
    print("  ✓ 理论故障频率计算")
    print("  ✓ 频率分析CSV文件")
    print("=" * 50)
    
    # 创建处理器
    processor = EnhancedBearingDataProcessor(base_path, output_path)
    
    # 处理所有文件
    segments = processor.process_all_files()
    
    print(f"\n处理完成！生成了 {len(segments)} 个数据片段")
    print("每个数据片段包含5个文件:")
    print("  - *_raw_data.npy: 去噪滤波后的原始数据片段")
    print("  - *_time_domain.png: 时域信号图像(带统计信息)")
    print("  - *_frequency_domain.png: 频域信号图像(带主频标注)")
    print("  - *_features.csv: 29维特征向量 + 故障类型标签")
    print("  - *_frequency_analysis.csv: 主频和谐波成分分析")

if __name__ == "__main__":
    main()
