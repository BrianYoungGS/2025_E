#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
源域数据集处理主程序 - 包含去噪滤波
实现降采样、去噪滤波、数据对齐、频域变换、特征提取的完整流程
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
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

class BearingDataProcessor:
    """轴承数据处理器"""
    
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
            'highpass_cutoff': 10,    # 高通滤波截止频率(Hz) - 去除低频噪声
            'lowpass_cutoff': 5000,   # 低通滤波截止频率(Hz) - 去除高频噪声
            'notch_freq': 50,         # 陷波滤波频率(Hz) - 去除工频干扰
            'notch_q': 30             # 陷波滤波品质因数
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
        
    def apply_denoising_filters(self, data, fs):
        """
        应用去噪滤波器组合
        
        Args:
            data: 输入信号
            fs: 采样频率
            
        Returns:
            filtered_data: 滤波后的信号
        """
        filtered_data = data.copy()
        
        # 1. 高通滤波 - 去除低频趋势和直流分量
        if self.filter_params['highpass_cutoff'] > 0:
            nyquist = fs / 2
            high_cutoff = self.filter_params['highpass_cutoff'] / nyquist
            if high_cutoff < 1.0:
                b_high, a_high = signal.butter(4, high_cutoff, btype='high')
                filtered_data = signal.filtfilt(b_high, a_high, filtered_data)
        
        # 2. 低通滤波 - 去除高频噪声
        if self.filter_params['lowpass_cutoff'] < fs/2:
            nyquist = fs / 2
            low_cutoff = self.filter_params['lowpass_cutoff'] / nyquist
            if low_cutoff < 1.0:
                b_low, a_low = signal.butter(4, low_cutoff, btype='low')
                filtered_data = signal.filtfilt(b_low, a_low, filtered_data)
        
        # 3. 陷波滤波 - 去除工频干扰(50Hz及其谐波)
        for harmonic in [1, 2, 3]:  # 50Hz, 100Hz, 150Hz
            notch_freq = self.filter_params['notch_freq'] * harmonic
            if notch_freq < fs/2:
                nyquist = fs / 2
                notch_norm = notch_freq / nyquist
                if 0 < notch_norm < 1:
                    b_notch, a_notch = signal.iirnotch(notch_norm, self.filter_params['notch_q'])
                    filtered_data = signal.filtfilt(b_notch, a_notch, filtered_data)
        
        # 4. 中值滤波 - 去除脉冲噪声
        if len(filtered_data) > 5:
            filtered_data = signal.medfilt(filtered_data, kernel_size=3)
        
        return filtered_data
    
    def downsample_data(self, data, original_fs, target_fs):
        """
        降采样数据
        
        Args:
            data: 原始数据
            original_fs: 原始采样频率
            target_fs: 目标采样频率
            
        Returns:
            downsampled_data: 降采样后的数据
        """
        if original_fs == target_fs:
            return data
            
        # 计算降采样比例
        downsample_ratio = original_fs / target_fs
        
        # 使用scipy的resample进行降采样（包含抗混叠滤波）
        new_length = int(len(data) / downsample_ratio)
        downsampled_data = signal.resample(data, new_length)
        
        return downsampled_data
    
    def extract_segments(self, data, segment_length):
        """
        从数据中提取前后两个片段
        
        Args:
            data: 输入数据
            segment_length: 片段长度
            
        Returns:
            front_segment, back_segment: 前段和后段数据
        """
        data_length = len(data)
        
        if data_length < segment_length * 2:
            # 如果数据太短，进行零填充
            padded_data = np.pad(data, (0, segment_length * 2 - data_length), 'constant')
            front_segment = padded_data[:segment_length]
            back_segment = padded_data[-segment_length:]
        else:
            front_segment = data[:segment_length]
            back_segment = data[-segment_length:]
            
        return front_segment, back_segment
    
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
        # 找到功率最大的频率
        max_power_idx = np.argmax(psd)
        features['P24_P25_MainFrequencyBandPosition1'] = frequencies[max_power_idx] if len(frequencies) > max_power_idx else 0
        
        # 找到第二大功率的频率
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
    
    def plot_time_domain(self, data, title, save_path):
        """绘制时域信号"""
        plt.figure(figsize=(12, 6))
        time_axis = np.arange(len(data)) / self.target_fs
        plt.plot(time_axis, data, 'b-', linewidth=0.8)
        plt.title(f'时域信号 - {title}')
        plt.xlabel('时间 (s)')
        plt.ylabel('幅值')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def plot_frequency_domain(self, magnitude, frequencies, title, save_path):
        """绘制频域信号"""
        plt.figure(figsize=(12, 6))
        plt.semilogy(frequencies, magnitude, 'r-', linewidth=0.8)
        plt.title(f'频域信号 - {title}')
        plt.xlabel('频率 (Hz)')
        plt.ylabel('幅值')
        plt.grid(True, alpha=0.3)
        plt.xlim(0, max(frequencies))
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def process_single_file(self, file_path):
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
            
            # 处理两个数据片段
            segments_info = []
            for i, (segment, suffix) in enumerate([(front_segment, 'front'), (back_segment, 'back')]):
                # 创建输出文件夹
                segment_name = f"{file_path.stem}_{suffix}_{self.segment_length}"
                segment_dir = self.output_path / "processed_segments" / segment_name
                segment_dir.mkdir(exist_ok=True)
                
                # 4. 特征提取
                time_features = self.calculate_time_domain_features(segment)
                freq_features, magnitude, frequencies = self.calculate_frequency_domain_features(segment, self.target_fs)
                
                # 合并所有特征
                all_features = {**time_features, **freq_features}
                all_features['fault_type'] = fault_type
                all_features['file_name'] = file_path.name
                all_features['segment'] = suffix
                
                # 5. 保存处理后的数据
                # 保存原始数据片段
                np.save(segment_dir / f"{segment_name}_raw_data.npy", segment)
                
                # 6. 绘制时域图
                time_plot_path = segment_dir / f"{segment_name}_time_domain.png"
                self.plot_time_domain(segment, f"{segment_name}", time_plot_path)
                
                # 7. 绘制频域图
                freq_plot_path = segment_dir / f"{segment_name}_frequency_domain.png"
                self.plot_frequency_domain(magnitude, frequencies, f"{segment_name}", freq_plot_path)
                
                # 8. 保存特征CSV
                features_df = pd.DataFrame([all_features])
                features_csv_path = segment_dir / f"{segment_name}_features.csv"
                features_df.to_csv(features_csv_path, index=False, encoding='utf-8-sig')
                
                segments_info.append({
                    'segment_name': segment_name,
                    'segment_dir': str(segment_dir),
                    'fault_type': fault_type,
                    'features': all_features
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
        categories = ['12kHz_DE_data', '12kHz_FE_data', '48kHz_DE_data', '48kHz_Normal_data']
        
        for category in categories:
            category_path = self.source_path / category
            if not category_path.exists():
                continue
                
            print(f"\n处理类别: {category}")
            
            # 递归查找所有.mat文件
            mat_files = list(category_path.rglob("*.mat"))
            total_files += len(mat_files)
            
            for file_path in mat_files:
                print(f"  处理文件: {file_path.relative_to(self.source_path)}")
                segments_info = self.process_single_file(file_path)
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
            'fault_type_distribution': {}
        }
        
        # 统计故障类型分布
        fault_types = [seg['fault_type'] for seg in self.processed_files]
        for fault_type in set(fault_types):
            report['fault_type_distribution'][fault_type] = fault_types.count(fault_type)
        
        # 保存JSON报告
        report_path = self.output_path / "reports" / "processing_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        # 生成汇总CSV
        all_features_list = []
        for seg in self.processed_files:
            features_row = seg['features'].copy()
            features_row['segment_name'] = seg['segment_name']
            all_features_list.append(features_row)
        
        if all_features_list:
            summary_df = pd.DataFrame(all_features_list)
            summary_csv_path = self.output_path / "reports" / "all_features_summary.csv"
            summary_df.to_csv(summary_csv_path, index=False, encoding='utf-8-sig')
        
        # 保存文本报告
        text_report_path = self.output_path / "reports" / "processing_report.txt"
        with open(text_report_path, 'w', encoding='utf-8') as f:
            f.write("源域数据处理报告\n")
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
            
            f.write(f"特征维度: {report['feature_count']} 个特征\n")
            f.write("特征列表: " + ", ".join(self.feature_names) + "\n")

def main():
    """主函数"""
    # 设置路径
    base_path = "/Users/gsyoung/MBP_documents/code/Matlab/研究生数学建模"
    output_path = "/Users/gsyoung/MBP_documents/code/Matlab/研究生数学建模/建模实现/处理后数据"
    
    print("源域数据处理器 - 包含去噪滤波")
    print("=" * 50)
    
    # 创建处理器
    processor = BearingDataProcessor(base_path, output_path)
    
    # 处理所有文件
    segments = processor.process_all_files()
    
    print(f"\n处理完成！生成了 {len(segments)} 个数据片段")
    print("每个数据片段包含:")
    print("  - raw_data.npy: 去噪滤波后的原始数据片段")
    print("  - time_domain.png: 时域信号图像")
    print("  - frequency_domain.png: 频域信号图像") 
    print("  - features.csv: 29维特征向量 + 故障类型标签")

if __name__ == "__main__":
    main()
