#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
重新设计的数据处理器
基于数据长度分析结果，从每个mat文件生成3个数据片段
确保每段>=6万点，降采样后>=2万点
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

class RedesignedDataProcessor:
    """重新设计的数据处理器"""
    
    def __init__(self, source_base_path, output_base_path):
        self.source_base = Path(source_base_path)
        self.output_base = Path(output_base_path)
        self.processed_count = 0
        
        # 数据长度分析结果（基于之前的分析）
        self.length_strategies = {
            '12kHz_DE_data': {
                'original_fs': 12000,
                'target_fs': 12000,
                'downsample_ratio': 1.0,
                'typical_length': 96000,
                'segments_per_file': 2,  # 96k只能分2段，每段48k
                'segment_length': 48000
            },
            '12kHz_FE_data': {
                'original_fs': 12000,
                'target_fs': 12000,
                'downsample_ratio': 1.0,
                'typical_length': 96000,
                'segments_per_file': 2,  # 96k只能分2段，每段48k
                'segment_length': 48000
            },
            '48kHz_DE_data': {
                'original_fs': 48000,
                'target_fs': 12000,
                'downsample_ratio': 4.0,
                'typical_length': 192000,
                'segments_per_file': 2,  # 192k降采样后48k，分2段每段24k
                'segment_length': 96000
            },
            '48kHz_Normal_data': {
                'original_fs': 48000,
                'target_fs': 12000,
                'downsample_ratio': 4.0,
                'typical_length': 350000,  # 平均长度
                'segments_per_file': 3,  # 只有这个可以分3段
                'segment_length': 116000  # 约116k每段
            }
        }
        
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
        # 先应用抗混叠滤波器，然后降采样
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
    
    def extract_multiple_segments(self, data, strategy):
        """从数据中提取多个片段"""
        data_length = len(data)
        segments_per_file = strategy['segments_per_file']
        
        # 如果数据长度不足，调整策略
        min_segment_length = 60000  # 最小6万点
        if data_length < min_segment_length * segments_per_file:
            # 数据不够，减少段数或调整段长度
            if data_length >= min_segment_length * 2:
                segments_per_file = 2
                segment_length = data_length // 2
            else:
                segments_per_file = 1
                segment_length = data_length
        else:
            segment_length = data_length // segments_per_file
        
        segments = []
        for i in range(segments_per_file):
            start_idx = i * segment_length
            end_idx = min(start_idx + segment_length, data_length)
            segment = data[start_idx:end_idx]
            
            # 确保段长度足够
            if len(segment) >= min_segment_length:
                segments.append(segment)
        
        return segments
    
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
        
        info_text = f'RMS: {rms:.4f}\nPeak: {peak:.4f}\nP-P: {peak2peak:.4f}'
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
    
    def process_single_file(self, file_path, category, file_index):
        """处理单个文件"""
        print(f"\n📄 处理文件: {file_path.name}")
        
        # 加载数据
        mat_data = self.load_mat_file(file_path)
        if mat_data is None:
            return []
        
        # 获取处理策略
        strategy = self.length_strategies.get(category)
        if strategy is None:
            print(f"❌ 未找到类别 {category} 的处理策略")
            return []
        
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
        
        # 处理每个时域变量
        for var_name in time_vars:
            data = mat_data[var_name]
            if not isinstance(data, np.ndarray) or len(data.shape) < 1:
                continue
            
            print(f"  处理变量: {var_name}, 长度: {len(data):,}")
            
            # 提取多个片段
            segments = self.extract_multiple_segments(data, strategy)
            
            for seg_idx, segment in enumerate(segments):
                segment_id = f"{file_path.stem}_{var_name.split('_')[-2]}_{seg_idx+1}"
                
                # 降采样
                if strategy['downsample_ratio'] > 1:
                    segment = self.downsample_signal(segment, strategy['original_fs'], strategy['target_fs'])
                
                # 去噪
                segment_denoised = self.apply_denoising(segment, strategy['target_fs'])
                
                # 特征提取
                time_features = self.extract_time_features(segment_denoised)
                freq_features, fft_magnitude, frequencies = self.compute_fft_features(segment_denoised, strategy['target_fs'])
                
                # 频率分析
                bearing_type = 'SKF6205' if 'DE' in var_name else 'SKF6203'
                freq_analysis = self.analyze_frequency_components(fft_magnitude, frequencies, strategy['target_fs'], rpm, bearing_type)
                
                # 确定故障类型
                fault_type = self._determine_fault_type(file_path.name, category)
                
                # 创建输出目录
                segment_dir = self.output_base / f"{segment_id}"
                segment_dir.mkdir(parents=True, exist_ok=True)
                
                # 保存原始数据
                np.save(segment_dir / f"{segment_id}_raw_data.npy", segment_denoised)
                
                # 绘制图像
                self.plot_time_domain(segment_denoised, strategy['target_fs'], 
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
                    'variable': var_name,
                    'segment_index': seg_idx,
                    'original_length': len(data),
                    'processed_length': len(segment_denoised),
                    'fault_type': fault_type,
                    'rpm': rpm,
                    'sampling_rate': strategy['target_fs']
                })
                
                self.processed_count += 1
                print(f"    ✅ 片段 {seg_idx+1}: {segment_id}")
        
        return processed_segments
    
    def _determine_fault_type(self, filename, category):
        """确定故障类型"""
        filename_upper = filename.upper()
        if 'OR' in filename_upper or 'OUTER' in filename_upper:
            return 'OR'
        elif 'IR' in filename_upper or 'INNER' in filename_upper:
            return 'IR'
        elif 'B' in filename_upper and ('007' in filename or '014' in filename or '021' in filename or '028' in filename):
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
    print("🚀 重新设计的数据处理器")
    print("=" * 60)
    print("目标: 从每个mat文件生成多个高质量数据片段")
    print("策略: 根据数据长度动态调整分段数量")
    print("=" * 60)
    
    # 路径设置
    source_base = "/Users/gsyoung/MBP_documents/code/Matlab/研究生数学建模/数据集/数据集/源域数据集"
    output_base = "/Users/gsyoung/MBP_documents/code/Matlab/研究生数学建模/建模实现/处理后数据/redesigned_segments"
    
    # 创建处理器
    processor = RedesignedDataProcessor(source_base, output_base)
    
    # 创建输出目录
    Path(output_base).mkdir(parents=True, exist_ok=True)
    
    # 删除旧的处理结果
    if Path(output_base).exists():
        import shutil
        shutil.rmtree(output_base)
        Path(output_base).mkdir(parents=True, exist_ok=True)
    
    all_processed_segments = []
    processing_stats = {}
    
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
        
        for file_idx, mat_file in enumerate(mat_files):
            segments = processor.process_single_file(mat_file, category, file_idx)
            category_segments.extend(segments)
            all_processed_segments.extend(segments)
        
        processing_stats[category] = {
            'input_files': len(mat_files),
            'output_segments': len(category_segments),
            'segments_per_file': len(category_segments) / len(mat_files) if len(mat_files) > 0 else 0
        }
        
        print(f"📊 {category} 统计:")
        print(f"  输入文件: {len(mat_files)} 个")
        print(f"  输出片段: {len(category_segments)} 个")
        print(f"  平均每文件片段数: {len(category_segments) / len(mat_files):.1f}")
    
    # 保存处理统计
    stats_summary = {
        'processing_time': datetime.now().isoformat(),
        'total_input_files': sum(stats['input_files'] for stats in processing_stats.values()),
        'total_output_segments': len(all_processed_segments),
        'category_stats': processing_stats,
        'segments_detail': all_processed_segments
    }
    
    # 保存到文件
    reports_dir = Path(output_base).parent / "reports"
    reports_dir.mkdir(exist_ok=True)
    
    with open(reports_dir / "redesigned_processing_report.json", 'w', encoding='utf-8') as f:
        json.dump(stats_summary, f, ensure_ascii=False, indent=2, default=str)
    
    # 保存详细统计CSV
    segments_df = pd.DataFrame(all_processed_segments)
    segments_df.to_csv(reports_dir / "redesigned_segments_summary.csv", index=False, encoding='utf-8-sig')
    
    print(f"\n✅ 处理完成!")
    print(f"📊 总结:")
    print(f"  输入文件: {stats_summary['total_input_files']} 个")
    print(f"  输出片段: {stats_summary['total_output_segments']} 个")
    print(f"  处理效率: {stats_summary['total_output_segments'] / stats_summary['total_input_files']:.1f} 片段/文件")
    print(f"📁 输出目录: {output_base}")


if __name__ == "__main__":
    main()
