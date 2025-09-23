#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
轴承故障诊断数据预处理模块
提供数据标准化、特征工程、数据增强等功能
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from scipy import signal
from scipy.stats import skew, kurtosis
from typing import Dict, List, Tuple, Optional, Union
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BearingDataPreprocessor:
    """
    轴承数据预处理器
    提供信号预处理、特征工程、数据标准化等功能
    """
    
    def __init__(self):
        self.scalers = {}
        self.feature_selectors = {}
        self.pca_transformers = {}
        
    def normalize_signals(self, 
                         signals: np.ndarray, 
                         method: str = 'standard',
                         fit_transform: bool = True) -> np.ndarray:
        """
        信号标准化
        
        Args:
            signals: 信号数组 (n_samples, signal_length)
            method: 标准化方法 'standard', 'minmax', 'robust'
            fit_transform: 是否拟合并转换
            
        Returns:
            标准化后的信号
        """
        if method not in self.scalers or fit_transform:
            if method == 'standard':
                scaler = StandardScaler()
            elif method == 'minmax':
                scaler = MinMaxScaler()
            elif method == 'robust':
                scaler = RobustScaler()
            else:
                raise ValueError(f"不支持的标准化方法: {method}")
            
            self.scalers[method] = scaler
            return scaler.fit_transform(signals)
        else:
            return self.scalers[method].transform(signals)
    
    def extract_time_domain_features(self, signals: np.ndarray) -> np.ndarray:
        """
        提取时域特征
        
        Args:
            signals: 信号数组 (n_samples, signal_length)
            
        Returns:
            时域特征数组 (n_samples, n_features)
        """
        features = []
        
        for signal in signals:
            # 基础统计特征
            mean_val = np.mean(signal)
            std_val = np.std(signal)
            var_val = np.var(signal)
            rms = np.sqrt(np.mean(signal**2))
            peak = np.max(np.abs(signal))
            
            # 形状特征
            skewness = skew(signal)
            kurt = kurtosis(signal)
            
            # 幅值特征
            peak_to_peak = np.ptp(signal)
            crest_factor = peak / rms if rms != 0 else 0
            clearance_factor = peak / np.mean(np.sqrt(np.abs(signal)))**2 if np.mean(np.sqrt(np.abs(signal))) != 0 else 0
            
            # 能量特征
            energy = np.sum(signal**2)
            entropy = -np.sum(signal**2 * np.log(signal**2 + 1e-8))
            
            feature_vector = [
                mean_val, std_val, var_val, rms, peak,
                skewness, kurt, peak_to_peak, crest_factor,
                clearance_factor, energy, entropy
            ]
            
            features.append(feature_vector)
        
        return np.array(features)
    
    def extract_frequency_domain_features(self, 
                                        signals: np.ndarray, 
                                        fs: float = 12000) -> np.ndarray:
        """
        提取频域特征
        
        Args:
            signals: 信号数组 (n_samples, signal_length)
            fs: 采样频率
            
        Returns:
            频域特征数组 (n_samples, n_features)
        """
        features = []
        
        for signal in signals:
            # 计算FFT
            fft_vals = np.fft.fft(signal)
            fft_mag = np.abs(fft_vals[:len(signal)//2])
            freqs = np.fft.fftfreq(len(signal), 1/fs)[:len(signal)//2]
            
            # 功率谱密度
            psd = fft_mag**2
            
            # 频域统计特征
            mean_freq = np.sum(freqs * psd) / np.sum(psd) if np.sum(psd) != 0 else 0
            freq_std = np.sqrt(np.sum((freqs - mean_freq)**2 * psd) / np.sum(psd)) if np.sum(psd) != 0 else 0
            
            # 频谱重心
            spectral_centroid = np.sum(freqs * fft_mag) / np.sum(fft_mag) if np.sum(fft_mag) != 0 else 0
            
            # 频谱带宽
            spectral_bandwidth = np.sqrt(np.sum((freqs - spectral_centroid)**2 * fft_mag) / np.sum(fft_mag)) if np.sum(fft_mag) != 0 else 0
            
            # 频谱偏度和峰度
            spectral_skewness = skew(fft_mag)
            spectral_kurtosis = kurtosis(fft_mag)
            
            # 主频成分
            dominant_freq = freqs[np.argmax(fft_mag)]
            max_magnitude = np.max(fft_mag)
            
            # 频域能量分布
            total_power = np.sum(psd)
            low_freq_power = np.sum(psd[freqs <= fs/6])
            mid_freq_power = np.sum(psd[(freqs > fs/6) & (freqs <= fs/3)])
            high_freq_power = np.sum(psd[freqs > fs/3])
            
            feature_vector = [
                mean_freq, freq_std, spectral_centroid, spectral_bandwidth,
                spectral_skewness, spectral_kurtosis, dominant_freq, max_magnitude,
                total_power, low_freq_power, mid_freq_power, high_freq_power
            ]
            
            features.append(feature_vector)
        
        return np.array(features)
    
    def extract_wavelet_features(self, 
                               signals: np.ndarray,
                               wavelet: str = 'db4',
                               levels: int = 4) -> np.ndarray:
        """
        提取小波特征
        
        Args:
            signals: 信号数组 (n_samples, signal_length)
            wavelet: 小波基函数
            levels: 分解层数
            
        Returns:
            小波特征数组 (n_samples, n_features)
        """
        try:
            import pywt
        except ImportError:
            logger.warning("PyWavelets未安装，跳过小波特征提取")
            return np.array([])
        
        features = []
        
        for signal in signals:
            # 小波分解
            coeffs = pywt.wavedec(signal, wavelet, level=levels)
            
            # 提取每层系数的统计特征
            feature_vector = []
            for coeff in coeffs:
                feature_vector.extend([
                    np.mean(np.abs(coeff)),
                    np.std(coeff),
                    np.max(np.abs(coeff)),
                    skew(coeff),
                    kurtosis(coeff)
                ])
            
            features.append(feature_vector)
        
        return np.array(features)
    
    def extract_comprehensive_features(self, 
                                     signals: np.ndarray,
                                     fs: float = 12000,
                                     include_wavelet: bool = True) -> np.ndarray:
        """
        提取综合特征
        
        Args:
            signals: 信号数组 (n_samples, signal_length)
            fs: 采样频率
            include_wavelet: 是否包含小波特征
            
        Returns:
            综合特征数组 (n_samples, n_features)
        """
        logger.info("开始提取综合特征...")
        
        # 时域特征
        time_features = self.extract_time_domain_features(signals)
        logger.info(f"时域特征: {time_features.shape}")
        
        # 频域特征
        freq_features = self.extract_frequency_domain_features(signals, fs)
        logger.info(f"频域特征: {freq_features.shape}")
        
        # 合并特征
        all_features = np.concatenate([time_features, freq_features], axis=1)
        
        # 小波特征（可选）
        if include_wavelet:
            wavelet_features = self.extract_wavelet_features(signals)
            if wavelet_features.size > 0:
                all_features = np.concatenate([all_features, wavelet_features], axis=1)
                logger.info(f"小波特征: {wavelet_features.shape}")
        
        logger.info(f"综合特征总维度: {all_features.shape}")
        return all_features
    
    def feature_selection(self, 
                         features: np.ndarray,
                         labels: np.ndarray,
                         method: str = 'univariate',
                         k: int = 50,
                         fit_transform: bool = True) -> np.ndarray:
        """
        特征选择
        
        Args:
            features: 特征数组
            labels: 标签数组
            method: 选择方法 'univariate', 'pca'
            k: 选择的特征数量
            fit_transform: 是否拟合并转换
            
        Returns:
            选择后的特征
        """
        if method == 'univariate':
            if method not in self.feature_selectors or fit_transform:
                selector = SelectKBest(f_classif, k=k)
                self.feature_selectors[method] = selector
                return selector.fit_transform(features, labels)
            else:
                return self.feature_selectors[method].transform(features)
                
        elif method == 'pca':
            if method not in self.pca_transformers or fit_transform:
                pca = PCA(n_components=k)
                self.pca_transformers[method] = pca
                return pca.fit_transform(features)
            else:
                return self.pca_transformers[method].transform(features)
        
        else:
            raise ValueError(f"不支持的特征选择方法: {method}")
    
    def augment_signals(self, 
                       signals: np.ndarray,
                       labels: np.ndarray,
                       methods: List[str] = ['noise', 'scaling'],
                       noise_level: float = 0.01,
                       scaling_range: Tuple[float, float] = (0.8, 1.2)) -> Tuple[np.ndarray, np.ndarray]:
        """
        信号数据增强
        
        Args:
            signals: 原始信号
            labels: 原始标签
            methods: 增强方法列表
            noise_level: 噪声水平
            scaling_range: 缩放范围
            
        Returns:
            增强后的信号和标签
        """
        augmented_signals = []
        augmented_labels = []
        
        # 保留原始数据
        augmented_signals.append(signals)
        augmented_labels.append(labels)
        
        for method in methods:
            if method == 'noise':
                # 添加高斯噪声
                noise = np.random.normal(0, noise_level, signals.shape)
                noisy_signals = signals + noise
                augmented_signals.append(noisy_signals)
                augmented_labels.append(labels)
                
            elif method == 'scaling':
                # 幅值缩放
                scale_factors = np.random.uniform(scaling_range[0], scaling_range[1], (len(signals), 1))
                scaled_signals = signals * scale_factors
                augmented_signals.append(scaled_signals)
                augmented_labels.append(labels)
                
            elif method == 'shift':
                # 时间偏移
                max_shift = signals.shape[1] // 10
                shifted_signals = []
                for signal in signals:
                    shift = np.random.randint(-max_shift, max_shift)
                    shifted_signal = np.roll(signal, shift)
                    shifted_signals.append(shifted_signal)
                augmented_signals.append(np.array(shifted_signals))
                augmented_labels.append(labels)
        
        # 合并所有增强数据
        all_signals = np.concatenate(augmented_signals, axis=0)
        all_labels = np.concatenate(augmented_labels, axis=0)
        
        logger.info(f"数据增强完成: {signals.shape[0]} -> {all_signals.shape[0]} 个样本")
        return all_signals, all_labels
    
    def balance_dataset(self, 
                       signals: np.ndarray,
                       labels: np.ndarray,
                       method: str = 'oversample') -> Tuple[np.ndarray, np.ndarray]:
        """
        数据集平衡
        
        Args:
            signals: 信号数据
            labels: 标签
            method: 平衡方法 'oversample', 'undersample'
            
        Returns:
            平衡后的数据和标签
        """
        unique_labels, counts = np.unique(labels, return_counts=True)
        logger.info(f"原始标签分布: {dict(zip(unique_labels, counts))}")
        
        if method == 'oversample':
            # 过采样到最大类别数量
            max_count = np.max(counts)
            balanced_signals = []
            balanced_labels = []
            
            for label in unique_labels:
                label_mask = labels == label
                label_signals = signals[label_mask]
                label_count = np.sum(label_mask)
                
                # 重复采样到目标数量
                if label_count < max_count:
                    indices = np.random.choice(len(label_signals), max_count, replace=True)
                    resampled_signals = label_signals[indices]
                else:
                    resampled_signals = label_signals
                
                balanced_signals.append(resampled_signals)
                balanced_labels.extend([label] * len(resampled_signals))
            
            balanced_signals = np.concatenate(balanced_signals, axis=0)
            balanced_labels = np.array(balanced_labels)
            
        elif method == 'undersample':
            # 欠采样到最小类别数量
            min_count = np.min(counts)
            balanced_signals = []
            balanced_labels = []
            
            for label in unique_labels:
                label_mask = labels == label
                label_signals = signals[label_mask]
                
                # 随机选择到目标数量
                indices = np.random.choice(len(label_signals), min_count, replace=False)
                resampled_signals = label_signals[indices]
                
                balanced_signals.append(resampled_signals)
                balanced_labels.extend([label] * len(resampled_signals))
            
            balanced_signals = np.concatenate(balanced_signals, axis=0)
            balanced_labels = np.array(balanced_labels)
        
        else:
            raise ValueError(f"不支持的平衡方法: {method}")
        
        unique_labels_new, counts_new = np.unique(balanced_labels, return_counts=True)
        logger.info(f"平衡后标签分布: {dict(zip(unique_labels_new, counts_new))}")
        
        return balanced_signals, balanced_labels

def create_preprocessing_pipeline(signals: np.ndarray,
                                labels: np.ndarray,
                                config: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    创建预处理流水线
    
    Args:
        signals: 原始信号
        labels: 标签
        config: 配置字典
        
    Returns:
        预处理后的信号和标签
    """
    preprocessor = BearingDataPreprocessor()
    
    # 信号标准化
    if config.get('normalize', True):
        signals = preprocessor.normalize_signals(
            signals, 
            method=config.get('normalize_method', 'standard')
        )
    
    # 数据增强
    if config.get('augment', False):
        signals, labels = preprocessor.augment_signals(
            signals, labels,
            methods=config.get('augment_methods', ['noise']),
            noise_level=config.get('noise_level', 0.01)
        )
    
    # 数据平衡
    if config.get('balance', False):
        signals, labels = preprocessor.balance_dataset(
            signals, labels,
            method=config.get('balance_method', 'oversample')
        )
    
    return signals, labels
