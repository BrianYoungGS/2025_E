#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
轴承故障诊断数据加载模块
用于读取renamed_segments和raw_data中的轴承振动数据，支持多模态数据融合

数据结构：
- renamed_segments/: 分段数据（483个文件夹）
- raw_data/: 完整原始数据（322个文件夹）

故障分类：N(正常), B(滚动体故障), IR(内圈故障), OR(外圈故障)
"""

import os
import re
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import cv2
from sklearn.preprocessing import LabelEncoder, StandardScaler
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BearingDataLoader:
    """
    轴承故障诊断数据加载器
    支持多模态数据（时域信号、频域信号、特征、图像）的统一加载和处理
    """
    
    def __init__(self, base_dir: str = None):
        """
        初始化数据加载器
        
        Args:
            base_dir: 基础数据目录路径
        """
        if base_dir is None:
            self.base_dir = Path("/Users/gsyoung/MBP_documents/code/Matlab/研究生数学建模/建模实现/处理后数据")
        else:
            self.base_dir = Path(base_dir)
            
        self.renamed_segments_dir = self.base_dir / "renamed_segments"
        self.raw_data_dir = self.base_dir / "raw_data"
        
        # 故障类型映射
        self.fault_types = {
            'N': 0,   # 正常
            'B': 1,   # 滚动体故障
            'IR': 2,  # 内圈故障  
            'OR': 3   # 外圈故障
        }
        
        # 初始化数据集信息
        self.segment_info = []
        self.raw_info = []
        self.label_encoder = LabelEncoder()
        self.feature_scaler = StandardScaler()
        
        logger.info(f"数据加载器初始化完成")
        logger.info(f"分段数据目录: {self.renamed_segments_dir}")
        logger.info(f"原始数据目录: {self.raw_data_dir}")
    
    def _parse_segment_filename(self, filename: str) -> Optional[Dict]:
        """
        解析分段数据文件名
        
        格式: {采样频率}_{传感器位置}_data_{故障类型}_{故障尺寸}_{载荷}_{片段编号}
        例如: 12kHz_DE_data_B_0007_0_1
        
        Args:
            filename: 文件夹名称
            
        Returns:
            包含解析信息的字典，解析失败返回None
        """
        # 正则表达式模式
        pattern = r'^(\d+kHz)_([A-Za-z]+)_data_([NBIRO]+)(?:_(\d{4}))?_(\d+)_(\d+)$'
        match = re.match(pattern, filename)
        
        if not match:
            logger.warning(f"无法解析文件名: {filename}")
            return None
        
        sampling_freq, sensor_type, fault_type, fault_size, load, segment_id = match.groups()
        
        return {
            'filename': filename,
            'sampling_freq': sampling_freq,
            'sensor_type': sensor_type,
            'fault_type': fault_type,
            'fault_size': fault_size if fault_size else '',
            'load': int(load),
            'segment_id': int(segment_id),
            'label': self.fault_types.get(fault_type, -1),
            'data_type': 'segment'
        }
    
    def _parse_raw_filename(self, filename: str) -> Optional[Dict]:
        """
        解析原始数据文件名
        
        格式: {采样频率}_{传感器位置}_{故障信息}[_(RPM)][_denoised]
        例如: 48k_DE_B007_0, 48k_Normal_N_0_denoised, 12k_DE_B028_0_(1797rpm)_denoised
        
        Args:
            filename: 文件夹名称
            
        Returns:
            包含解析信息的字典，解析失败返回None
        """
        # 检查是否为去噪版本
        is_denoised = filename.endswith('_denoised')
        if is_denoised:
            filename_clean = filename[:-9]  # 移除 '_denoised'
        else:
            filename_clean = filename
        
        # 正则表达式模式
        patterns = [
            # 带RPM的故障数据: 12k_DE_B028_0_(1797rpm)
            r'^(\d+k)_([A-Za-z]+)_([BIR]+)(\d{3})_(\d+)_\(.+rpm\)$',
            # 带RPM的外圈故障: 48k_DE_OR007@3_0_(1797rpm)
            r'^(\d+k)_([A-Za-z]+)_(OR)(\d{3})@\d+_(\d+)_\(.+rpm\)$',
            # 带RPM的正常数据: 48k_Normal_N_0_(1797rpm)
            r'^(\d+k)_([A-Za-z]+)_(N)_(\d+)_\(.+rpm\)$',
            # 普通故障数据: 48k_DE_B007_0
            r'^(\d+k)_([A-Za-z]+)_([BIR]+)(\d{3})_(\d+)$',
            # 普通外圈故障: 48k_DE_OR007@3_0  
            r'^(\d+k)_([A-Za-z]+)_(OR)(\d{3})@\d+_(\d+)$',
            # 普通正常数据: 48k_Normal_N_0
            r'^(\d+k)_([A-Za-z]+)_(N)_(\d+)$'
        ]
        
        for pattern in patterns:
            match = re.match(pattern, filename_clean)
            if match:
                groups = match.groups()
                
                # 根据匹配的组数判断数据类型
                if len(groups) == 5:
                    sampling_freq, sensor_type, fault_type, fault_info, load = groups
                    if fault_type in ['B', 'IR', 'OR']:
                        fault_size = fault_info
                    else:  # fault_type == 'N'，此时fault_info实际是load
                        fault_size = ''
                        load = fault_info
                elif len(groups) == 4:  # 正常数据情况
                    sampling_freq, sensor_type, fault_type, load = groups
                    fault_size = ''
                else:
                    continue  # 跳过无法处理的格式
                
                return {
                    'filename': filename,
                    'sampling_freq': sampling_freq + 'Hz',  # 统一格式
                    'sensor_type': sensor_type,
                    'fault_type': fault_type,
                    'fault_size': fault_size,
                    'load': int(load),
                    'is_denoised': is_denoised,
                    'label': self.fault_types.get(fault_type, -1),
                    'data_type': 'raw'
                }
        
        logger.warning(f"无法解析原始数据文件名: {filename}")
        return None
    
    def scan_datasets(self) -> Tuple[List[Dict], List[Dict]]:
        """
        扫描数据集，获取所有可用数据的元信息
        
        Returns:
            (segment_info, raw_info): 分段数据信息列表和原始数据信息列表
        """
        logger.info("开始扫描数据集...")
        
        # 扫描分段数据
        segment_info = []
        if self.renamed_segments_dir.exists():
            for folder in self.renamed_segments_dir.iterdir():
                if folder.is_dir():
                    info = self._parse_segment_filename(folder.name)
                    if info:
                        info['folder_path'] = folder
                        segment_info.append(info)
        
        # 扫描原始数据
        raw_info = []
        if self.raw_data_dir.exists():
            for folder in self.raw_data_dir.iterdir():
                if folder.is_dir():
                    info = self._parse_raw_filename(folder.name)
                    if info:
                        info['folder_path'] = folder
                        raw_info.append(info)
        
        self.segment_info = segment_info
        self.raw_info = raw_info
        
        logger.info(f"扫描完成: 分段数据 {len(segment_info)} 个, 原始数据 {len(raw_info)} 个")
        return segment_info, raw_info
    
    def load_sample_data(self, folder_path: Path, data_info: Dict) -> Dict:
        """
        加载单个样本的所有模态数据
        
        Args:
            folder_path: 数据文件夹路径
            data_info: 数据信息字典
            
        Returns:
            包含所有模态数据的字典
        """
        sample_data = {
            'info': data_info,
            'raw_signal': None,
            'features': None,
            'frequency_analysis': None,
            'time_domain_image': None,
            'frequency_domain_image': None
        }
        
        folder_name = folder_path.name
        
        try:
            # 1. 加载原始信号数据 (.npy)
            raw_file = folder_path / f"{folder_name}_raw_data.npy"
            if raw_file.exists():
                sample_data['raw_signal'] = np.load(raw_file)
            
            # 2. 加载特征数据 (.csv)
            features_file = folder_path / f"{folder_name}_features.csv"
            if features_file.exists():
                sample_data['features'] = pd.read_csv(features_file)
            
            # 3. 加载频域分析数据 (.csv)
            freq_analysis_file = folder_path / f"{folder_name}_frequency_analysis.csv"
            if freq_analysis_file.exists():
                sample_data['frequency_analysis'] = pd.read_csv(freq_analysis_file)
            
            # 4. 加载时域图像 (.png)
            time_img_file = folder_path / f"{folder_name}_time_domain.png"
            if time_img_file.exists():
                sample_data['time_domain_image'] = cv2.imread(str(time_img_file))
            
            # 5. 加载频域图像 (.png)
            freq_img_file = folder_path / f"{folder_name}_frequency_domain.png"
            if freq_img_file.exists():
                sample_data['frequency_domain_image'] = cv2.imread(str(freq_img_file))
                
        except Exception as e:
            logger.error(f"加载数据失败 {folder_path}: {e}")
        
        return sample_data
    
    def filter_by_criteria(self, 
                          fault_types: List[str] = None,
                          sampling_freqs: List[str] = None,
                          sensor_types: List[str] = None,
                          loads: List[int] = None,
                          data_type: str = 'both',
                          denoised_only: bool = False) -> List[Dict]:
        """
        根据条件筛选数据
        
        Args:
            fault_types: 故障类型列表 ['N', 'B', 'IR', 'OR']
            sampling_freqs: 采样频率列表 ['12kHz', '48kHz']
            sensor_types: 传感器类型列表 ['DE', 'FE', 'Normal']
            loads: 载荷列表 [0, 1, 2, 3]
            data_type: 数据类型 'segment', 'raw', 'both'
            denoised_only: 仅返回去噪数据（仅对raw_data有效）
            
        Returns:
            符合条件的数据信息列表
        """
        if not self.segment_info and not self.raw_info:
            self.scan_datasets()
        
        filtered_data = []
        
        # 处理分段数据
        if data_type in ['segment', 'both']:
            for info in self.segment_info:
                if fault_types and info['fault_type'] not in fault_types:
                    continue
                if sampling_freqs and info['sampling_freq'] not in sampling_freqs:
                    continue
                if sensor_types and info['sensor_type'] not in sensor_types:
                    continue
                if loads and info['load'] not in loads:
                    continue
                filtered_data.append(info)
        
        # 处理原始数据
        if data_type in ['raw', 'both']:
            for info in self.raw_info:
                if fault_types and info['fault_type'] not in fault_types:
                    continue
                if sampling_freqs and info['sampling_freq'] not in sampling_freqs:
                    continue
                if sensor_types and info['sensor_type'] not in sensor_types:
                    continue
                if loads and info['load'] not in loads:
                    continue
                if denoised_only and not info['is_denoised']:
                    continue
                filtered_data.append(info)
        
        logger.info(f"筛选结果: {len(filtered_data)} 个样本符合条件")
        return filtered_data
    
    def load_dataset(self, 
                    data_list: List[Dict],
                    include_raw: bool = True,
                    include_features: bool = True,
                    include_freq_analysis: bool = True,
                    include_images: bool = False,
                    max_samples: int = None) -> Dict:
        """
        批量加载数据集
        
        Args:
            data_list: 数据信息列表
            include_raw: 是否包含原始信号
            include_features: 是否包含特征数据
            include_freq_analysis: 是否包含频域分析
            include_images: 是否包含图像数据
            max_samples: 最大样本数量限制
            
        Returns:
            包含所有数据的字典
        """
        if max_samples:
            data_list = data_list[:max_samples]
        
        dataset = {
            'raw_signals': [],
            'features': [],
            'frequency_analysis': [],
            'time_images': [],
            'frequency_images': [],
            'labels': [],
            'info': []
        }
        
        logger.info(f"开始加载 {len(data_list)} 个样本...")
        
        for i, data_info in enumerate(data_list):
            if i % 50 == 0:
                logger.info(f"已加载 {i}/{len(data_list)} 个样本")
            
            sample = self.load_sample_data(data_info['folder_path'], data_info)
            
            # 添加到数据集
            if include_raw and sample['raw_signal'] is not None:
                dataset['raw_signals'].append(sample['raw_signal'])
            
            if include_features and sample['features'] is not None:
                dataset['features'].append(sample['features'].values.flatten())
            
            if include_freq_analysis and sample['frequency_analysis'] is not None:
                dataset['frequency_analysis'].append(sample['frequency_analysis'].values.flatten())
            
            if include_images:
                if sample['time_domain_image'] is not None:
                    dataset['time_images'].append(sample['time_domain_image'])
                if sample['frequency_domain_image'] is not None:
                    dataset['frequency_images'].append(sample['frequency_domain_image'])
            
            dataset['labels'].append(data_info['label'])
            dataset['info'].append(data_info)
        
        # 转换为numpy数组
        for key in ['raw_signals', 'features', 'frequency_analysis', 'labels']:
            if dataset[key]:
                dataset[key] = np.array(dataset[key])
        
        logger.info(f"数据加载完成: {len(dataset['labels'])} 个样本")
        return dataset
    
    def get_dataset_statistics(self) -> Dict:
        """
        获取数据集统计信息
        
        Returns:
            数据集统计信息字典
        """
        if not self.segment_info and not self.raw_info:
            self.scan_datasets()
        
        stats = {
            'total_samples': len(self.segment_info) + len(self.raw_info),
            'segment_samples': len(self.segment_info),
            'raw_samples': len(self.raw_info),
            'fault_distribution': {},
            'sampling_freq_distribution': {},
            'sensor_type_distribution': {},
            'load_distribution': {}
        }
        
        all_data = self.segment_info + self.raw_info
        
        # 统计分布
        for data in all_data:
            # 故障类型分布
            fault_type = data['fault_type']
            stats['fault_distribution'][fault_type] = stats['fault_distribution'].get(fault_type, 0) + 1
            
            # 采样频率分布
            freq = data['sampling_freq']
            stats['sampling_freq_distribution'][freq] = stats['sampling_freq_distribution'].get(freq, 0) + 1
            
            # 传感器类型分布
            sensor = data['sensor_type']
            stats['sensor_type_distribution'][sensor] = stats['sensor_type_distribution'].get(sensor, 0) + 1
            
            # 载荷分布
            load = data['load']
            stats['load_distribution'][load] = stats['load_distribution'].get(load, 0) + 1
        
        return stats

# 便捷函数
def create_train_test_split(data_loader: BearingDataLoader, 
                           test_ratio: float = 0.2,
                           stratify_by: str = 'fault_type',
                           **filter_kwargs) -> Tuple[Dict, Dict]:
    """
    创建训练和测试数据集
    
    Args:
        data_loader: 数据加载器实例
        test_ratio: 测试集比例
        stratify_by: 分层采样的依据
        **filter_kwargs: 数据筛选参数
        
    Returns:
        (train_dataset, test_dataset): 训练和测试数据集
    """
    from sklearn.model_selection import train_test_split
    
    # 获取符合条件的数据
    filtered_data = data_loader.filter_by_criteria(**filter_kwargs)
    
    # 准备分层标签
    if stratify_by == 'fault_type':
        stratify_labels = [data['fault_type'] for data in filtered_data]
    else:
        stratify_labels = None
    
    # 分割数据
    train_data, test_data = train_test_split(
        filtered_data, 
        test_size=test_ratio,
        stratify=stratify_labels,
        random_state=42
    )
    
    logger.info(f"数据分割完成: 训练集 {len(train_data)} 个, 测试集 {len(test_data)} 个")
    
    # 加载数据
    train_dataset = data_loader.load_dataset(train_data)
    test_dataset = data_loader.load_dataset(test_data)
    
    return train_dataset, test_dataset

def demo_usage():
    """
    数据加载器使用示例
    """
    # 创建数据加载器
    loader = BearingDataLoader()
    
    # 扫描数据集
    loader.scan_datasets()
    
    # 获取统计信息
    stats = loader.get_dataset_statistics()
    print("=== 数据集统计信息 ===")
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # 筛选特定条件的数据
    filtered_data = loader.filter_by_criteria(
        fault_types=['N', 'B', 'IR', 'OR'],  # 所有故障类型
        sampling_freqs=['12kHz'],             # 仅12kHz数据
        sensor_types=['DE'],                  # 仅驱动端数据
        data_type='segment'                   # 仅分段数据
    )
    print(f"\n筛选到 {len(filtered_data)} 个样本")
    
    # 加载数据集（小样本测试）
    dataset = loader.load_dataset(
        filtered_data[:10],  # 仅加载前10个样本
        include_raw=True,
        include_features=True,
        include_images=False
    )
    
    print(f"\n加载的数据形状:")
    if len(dataset['raw_signals']) > 0:
        print(f"原始信号: {dataset['raw_signals'].shape}")
    if len(dataset['features']) > 0:
        print(f"特征数据: {dataset['features'].shape}")
    print(f"标签: {dataset['labels'].shape}")

if __name__ == "__main__":
    demo_usage()
