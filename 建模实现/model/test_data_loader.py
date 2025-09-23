#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化版数据加载器测试
不依赖numpy等库，仅测试基本的文件扫描和解析功能
"""

import os
import re
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleBearingDataLoader:
    """
    简化版轴承数据加载器
    仅用于测试文件扫描和解析功能
    """
    
    def __init__(self, base_dir=None):
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
        
        logger.info(f"数据加载器初始化完成")
        logger.info(f"分段数据目录: {self.renamed_segments_dir}")
        logger.info(f"原始数据目录: {self.raw_data_dir}")
    
    def _parse_segment_filename(self, filename):
        """解析分段数据文件名"""
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
    
    def _parse_raw_filename(self, filename):
        """解析原始数据文件名"""
        is_denoised = filename.endswith('_denoised')
        if is_denoised:
            filename_clean = filename[:-9]
        else:
            filename_clean = filename
        
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
                
                if len(groups) == 5:
                    sampling_freq, sensor_type, fault_type, fault_info, load = groups
                    if fault_type in ['B', 'IR', 'OR']:
                        fault_size = fault_info
                    else:  # fault_type == 'N'
                        fault_size = ''
                        load = fault_info
                elif len(groups) == 4:
                    sampling_freq, sensor_type, fault_type, load = groups
                    fault_size = ''
                else:
                    continue
                
                return {
                    'filename': filename,
                    'sampling_freq': sampling_freq + 'Hz',
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
    
    def scan_datasets(self):
        """扫描数据集"""
        logger.info("开始扫描数据集...")
        
        segment_info = []
        if self.renamed_segments_dir.exists():
            for folder in self.renamed_segments_dir.iterdir():
                if folder.is_dir():
                    info = self._parse_segment_filename(folder.name)
                    if info:
                        info['folder_path'] = folder
                        segment_info.append(info)
        
        raw_info = []
        if self.raw_data_dir.exists():
            for folder in self.raw_data_dir.iterdir():
                if folder.is_dir():
                    info = self._parse_raw_filename(folder.name)
                    if info:
                        info['folder_path'] = folder
                        raw_info.append(info)
        
        logger.info(f"扫描完成: 分段数据 {len(segment_info)} 个, 原始数据 {len(raw_info)} 个")
        return segment_info, raw_info
    
    def filter_by_criteria(self, segment_info, raw_info, **kwargs):
        """根据条件筛选数据"""
        fault_types = kwargs.get('fault_types')
        sampling_freqs = kwargs.get('sampling_freqs')
        sensor_types = kwargs.get('sensor_types')
        loads = kwargs.get('loads')
        data_type = kwargs.get('data_type', 'both')
        
        filtered_data = []
        
        if data_type in ['segment', 'both']:
            for info in segment_info:
                if fault_types and info['fault_type'] not in fault_types:
                    continue
                if sampling_freqs and info['sampling_freq'] not in sampling_freqs:
                    continue
                if sensor_types and info['sensor_type'] not in sensor_types:
                    continue
                if loads and info['load'] not in loads:
                    continue
                filtered_data.append(info)
        
        if data_type in ['raw', 'both']:
            for info in raw_info:
                if fault_types and info['fault_type'] not in fault_types:
                    continue
                if sampling_freqs and info['sampling_freq'] not in sampling_freqs:
                    continue
                if sensor_types and info['sensor_type'] not in sensor_types:
                    continue
                if loads and info['load'] not in loads:
                    continue
                filtered_data.append(info)
        
        logger.info(f"筛选结果: {len(filtered_data)} 个样本符合条件")
        return filtered_data
    
    def get_statistics(self, segment_info, raw_info):
        """获取统计信息"""
        all_data = segment_info + raw_info
        
        stats = {
            'total_samples': len(all_data),
            'segment_samples': len(segment_info),
            'raw_samples': len(raw_info),
            'fault_distribution': {},
            'sampling_freq_distribution': {},
            'sensor_type_distribution': {},
            'load_distribution': {}
        }
        
        for data in all_data:
            fault_type = data['fault_type']
            stats['fault_distribution'][fault_type] = stats['fault_distribution'].get(fault_type, 0) + 1
            
            freq = data['sampling_freq']
            stats['sampling_freq_distribution'][freq] = stats['sampling_freq_distribution'].get(freq, 0) + 1
            
            sensor = data['sensor_type']
            stats['sensor_type_distribution'][sensor] = stats['sensor_type_distribution'].get(sensor, 0) + 1
            
            load = data['load']
            stats['load_distribution'][load] = stats['load_distribution'].get(load, 0) + 1
        
        return stats

def test_data_loading():
    """测试数据加载功能"""
    print("=" * 60)
    print("轴承故障诊断数据加载测试")
    print("=" * 60)
    
    # 创建数据加载器
    loader = SimpleBearingDataLoader()
    
    # 扫描数据集
    segment_info, raw_info = loader.scan_datasets()
    
    # 获取统计信息
    stats = loader.get_statistics(segment_info, raw_info)
    
    print("\n📊 数据集统计信息:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # 测试文件名解析
    print("\n🔍 文件名解析测试:")
    test_filenames = [
        "12kHz_DE_data_B_0007_0_1",
        "12kHz_FE_data_IR_0021_2_456",
        "48kHz_Normal_data_N_0_481",
        "12kHz_DE_data_OR_0014_1_367"
    ]
    
    for filename in test_filenames:
        parsed = loader._parse_segment_filename(filename)
        if parsed:
            print(f"  ✅ {filename}")
            print(f"     -> 故障类型: {parsed['fault_type']}, 频率: {parsed['sampling_freq']}, 传感器: {parsed['sensor_type']}")
        else:
            print(f"  ❌ {filename} (解析失败)")
    
    # 测试数据筛选
    print("\n🎯 数据筛选测试:")
    
    filter_configs = [
        {
            'name': '12kHz DE数据',
            'filter': {
                'sampling_freqs': ['12kHz'],
                'sensor_types': ['DE'],
                'data_type': 'segment'
            }
        },
        {
            'name': '滚动体故障数据',
            'filter': {
                'fault_types': ['B'],
                'data_type': 'both'
            }
        },
        {
            'name': '48kHz数据',
            'filter': {
                'sampling_freqs': ['48kHz'],
                'data_type': 'raw'
            }
        }
    ]
    
    for config in filter_configs:
        filtered = loader.filter_by_criteria(segment_info, raw_info, **config['filter'])
        print(f"  {config['name']}: {len(filtered)} 个样本")
        
        # 显示故障类型分布
        fault_dist = {}
        for data in filtered:
            fault = data['fault_type']
            fault_dist[fault] = fault_dist.get(fault, 0) + 1
        print(f"    故障分布: {fault_dist}")
    
    # 检查文件内容
    print("\n📁 文件内容检查:")
    if segment_info:
        sample_folder = segment_info[0]['folder_path']
        print(f"  检查样本文件夹: {sample_folder.name}")
        
        expected_files = [
            f"{sample_folder.name}_features.csv",
            f"{sample_folder.name}_frequency_analysis.csv",
            f"{sample_folder.name}_frequency_domain.png",
            f"{sample_folder.name}_raw_data.npy",
            f"{sample_folder.name}_time_domain.png"
        ]
        
        for expected_file in expected_files:
            file_path = sample_folder / expected_file
            if file_path.exists():
                file_size = file_path.stat().st_size
                print(f"    ✅ {expected_file} ({file_size} bytes)")
            else:
                print(f"    ❌ {expected_file} (缺失)")
    
    print("\n✅ 数据加载测试完成！")
    print("=" * 60)
    
    return stats

if __name__ == "__main__":
    test_data_loading()
