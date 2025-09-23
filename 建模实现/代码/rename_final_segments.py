#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
轴承数据片段重命名程序
将final_segments中的数据片段重新命名，使其符合原始数据集的命名规范

命名规范：{采样频率}_{传感器位置}_data_{故障类型}_{故障尺寸}_{载荷}_{片段编号}
示例：12kHz_DE_data_B_0007_0_1
"""

import os
import re
import shutil
from pathlib import Path
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SegmentRenamer:
    def __init__(self):
        self.source_segments_dir = Path("/Users/gsyoung/MBP_documents/code/Matlab/研究生数学建模/建模实现/处理后数据/final_segments")
        self.source_dataset_dir = Path("/Users/gsyoung/MBP_documents/code/Matlab/研究生数学建模/数据集/数据集/源域数据集")
        self.output_dir = Path("/Users/gsyoung/MBP_documents/code/Matlab/研究生数学建模/建模实现/处理后数据/renamed_segments")
        
        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 建立原始数据集的映射关系
        self.dataset_mapping = self._build_dataset_mapping()
        
        # 建立当前片段的解析规则
        self.current_segments = self._analyze_current_segments()
        
    def _build_dataset_mapping(self):
        """
        构建原始数据集的映射关系
        返回：{原始文件名: {category, sampling_freq, sensor_type, fault_type, fault_size, load}}
        """
        mapping = {}
        
        # 遍历源域数据集
        for freq_dir in self.source_dataset_dir.iterdir():
            if not freq_dir.is_dir():
                continue
                
            freq_name = freq_dir.name  # 12kHz_DE_data, 48kHz_Normal_data, etc.
            
            # 解析频率和传感器类型
            if freq_name.startswith('12kHz_'):
                sampling_freq = '12kHz'
                if 'DE_data' in freq_name:
                    sensor_type = 'DE'
                elif 'FE_data' in freq_name:
                    sensor_type = 'FE'
                elif 'Normal_data' in freq_name:
                    sensor_type = 'Normal'
                else:
                    sensor_type = 'Unknown'
            elif freq_name.startswith('48kHz_'):
                sampling_freq = '48kHz'
                if 'DE_data' in freq_name:
                    sensor_type = 'DE'
                elif 'Normal_data' in freq_name:
                    sensor_type = 'Normal'
                else:
                    sensor_type = 'Unknown'
            else:
                continue
                
            if sensor_type == 'Normal':
                # 处理正常数据
                for mat_file in freq_dir.glob('*.mat'):
                    file_key = mat_file.stem  # N_0, N_1_(1772rpm), etc.
                    mapping[file_key] = {
                        'category': freq_name,
                        'sampling_freq': sampling_freq,
                        'sensor_type': sensor_type,
                        'fault_type': 'N',
                        'fault_size': '',
                        'load': self._extract_normal_load(file_key)
                    }
            else:
                # 处理故障数据
                for fault_type_dir in freq_dir.iterdir():
                    if not fault_type_dir.is_dir():
                        continue
                        
                    fault_type = fault_type_dir.name  # B, IR, OR
                    
                    for size_dir in fault_type_dir.iterdir():
                        if not size_dir.is_dir():
                            continue
                            
                        fault_size = size_dir.name  # 0007, 0014, 0021, 0028
                        
                        for mat_file in size_dir.glob('*.mat'):
                            file_key = mat_file.stem  # B007_0, IR014_1, etc.
                            load = file_key.split('_')[-1]  # 0, 1, 2, 3
                            
                            mapping[file_key] = {
                                'category': freq_name,
                                'sampling_freq': sampling_freq,
                                'sensor_type': sensor_type,
                                'fault_type': fault_type,
                                'fault_size': fault_size,
                                'load': load
                            }
                            
        logger.info(f"构建数据集映射关系完成，共 {len(mapping)} 个原始文件")
        return mapping
    
    def _extract_normal_load(self, filename):
        """提取正常数据的载荷信息"""
        if filename == 'N_0':
            return '0'
        elif filename.startswith('N_1'):
            return '1'
        elif filename.startswith('N_2'):
            return '2'
        elif filename == 'N_3':
            return '3'
        else:
            return '0'
    
    def _analyze_current_segments(self):
        """
        分析当前final_segments中的文件夹命名模式
        返回：{当前文件夹名: {原始文件名, 片段编号}}
        """
        segments = {}
        
        for segment_dir in self.source_segments_dir.iterdir():
            if not segment_dir.is_dir():
                continue
                
            dir_name = segment_dir.name
            # 解析格式如：N_0_1_481, B007_0_3_453, OR007@3_2_1_433
            
            # 使用正则表达式解析
            # 模式1: N_载荷_片段_编号 (简单正常数据)
            pattern1 = r'^N_(\d+)_(\d+)_(\d+)$'
            match1 = re.match(pattern1, dir_name)
            if match1:
                load, segment_num, _ = match1.groups()
                original_file = f"N_{load}"
                if load == '1':
                    original_file = "N_1_(1772rpm)"
                elif load == '2':
                    original_file = "N_2_(1750rpm)"
                segments[dir_name] = {
                    'original_file': original_file,
                    'segment_num': segment_num
                }
                continue
                
            # 模式1b: N_载荷_(转速)_片段_编号 (带转速的正常数据)
            pattern1b = r'^N_(\d+)_\(.+\)_(\d+)_(\d+)$'
            match1b = re.match(pattern1b, dir_name)
            if match1b:
                load, segment_num, _ = match1b.groups()
                if load == '1':
                    original_file = "N_1_(1772rpm)"
                elif load == '2':
                    original_file = "N_2_(1750rpm)"
                else:
                    original_file = f"N_{load}"
                segments[dir_name] = {
                    'original_file': original_file,
                    'segment_num': segment_num
                }
                continue
            
            # 模式2: 故障类型+尺寸_载荷_片段_编号 (简单B/IR故障)
            pattern2 = r'^([BIR]+)(\d{3})_(\d+)_(\d+)_(\d+)$'
            match2 = re.match(pattern2, dir_name)
            if match2:
                fault_type, size, load, segment_num, _ = match2.groups()
                original_file = f"{fault_type}{size}_{load}"
                segments[dir_name] = {
                    'original_file': original_file,
                    'segment_num': segment_num
                }
                continue
                
            # 模式2b: 故障类型+尺寸_载荷_(转速)_片段_编号 (带转速的B/IR故障)
            pattern2b = r'^([BIR]+)(\d{3})_(\d+)_\(.+\)_(\d+)_(\d+)$'
            match2b = re.match(pattern2b, dir_name)
            if match2b:
                fault_type, size, load, segment_num, _ = match2b.groups()
                original_file = f"{fault_type}{size}_{load}"
                segments[dir_name] = {
                    'original_file': original_file,
                    'segment_num': segment_num
                }
                continue
            
            # 模式3: OR故障 (OR007@3_2_1_433)
            pattern3 = r'^OR(\d{3})@(\d+)_(\d+)_(\d+)_(\d+)$'
            match3 = re.match(pattern3, dir_name)
            if match3:
                size, position, load, segment_num, _ = match3.groups()
                original_file = f"OR{size}@{position}_{load}"
                segments[dir_name] = {
                    'original_file': original_file,
                    'segment_num': segment_num
                }
                continue
                
            logger.warning(f"无法解析文件夹名称: {dir_name}")
            
        logger.info(f"分析当前片段完成，共 {len(segments)} 个片段")
        return segments
    
    def _generate_new_name(self, original_file, segment_num):
        """
        根据原始文件名和片段编号生成新的标准命名
        """
        if original_file not in self.dataset_mapping:
            logger.error(f"找不到原始文件映射: {original_file}")
            return None
            
        info = self.dataset_mapping[original_file]
        
        # 构建新名称
        parts = [
            info['sampling_freq'],      # 12kHz, 48kHz
            info['sensor_type'],        # DE, FE, Normal
            'data',                     # 固定
            info['fault_type'],         # B, IR, OR, N
        ]
        
        # 添加故障尺寸（如果有）
        if info['fault_size']:
            parts.append(info['fault_size'])
            
        # 添加载荷
        parts.append(info['load'])
        
        # 添加片段编号
        parts.append(segment_num)
        
        return '_'.join(parts)
    
    def rename_segments(self):
        """
        执行批量重命名
        """
        rename_log = []
        success_count = 0
        error_count = 0
        
        for current_name, info in self.current_segments.items():
            original_file = info['original_file']
            segment_num = info['segment_num']
            
            # 生成新名称
            new_name = self._generate_new_name(original_file, segment_num)
            if not new_name:
                error_count += 1
                continue
                
            # 执行重命名
            old_path = self.source_segments_dir / current_name
            new_path = self.output_dir / new_name
            
            try:
                # 复制整个文件夹
                shutil.copytree(old_path, new_path)
                
                # 重命名文件夹内的文件
                self._rename_internal_files(new_path, current_name, new_name)
                
                rename_log.append({
                    'old_name': current_name,
                    'new_name': new_name,
                    'original_file': original_file,
                    'segment_num': segment_num,
                    'status': 'success'
                })
                success_count += 1
                
                if success_count % 50 == 0:
                    logger.info(f"已处理 {success_count} 个文件夹")
                    
            except Exception as e:
                logger.error(f"重命名失败: {current_name} -> {new_name}, 错误: {e}")
                rename_log.append({
                    'old_name': current_name,
                    'new_name': new_name,
                    'original_file': original_file,
                    'segment_num': segment_num,
                    'status': 'error',
                    'error': str(e)
                })
                error_count += 1
        
        # 保存重命名日志
        self._save_rename_log(rename_log)
        
        logger.info(f"重命名完成！成功: {success_count}, 失败: {error_count}")
        return rename_log
    
    def _rename_internal_files(self, folder_path, old_prefix, new_prefix):
        """
        重命名文件夹内的文件
        """
        for file_path in folder_path.iterdir():
            if file_path.is_file():
                old_name = file_path.name
                # 替换文件名前缀
                new_name = old_name.replace(old_prefix, new_prefix)
                new_file_path = folder_path / new_name
                file_path.rename(new_file_path)
    
    def _save_rename_log(self, rename_log):
        """
        保存重命名日志
        """
        import pandas as pd
        
        df = pd.DataFrame(rename_log)
        log_file = self.output_dir.parent / 'rename_log.csv'
        df.to_csv(log_file, index=False, encoding='utf-8-sig')
        
        # 生成总结报告
        summary = {
            'total_files': len(rename_log),
            'success_count': len([x for x in rename_log if x['status'] == 'success']),
            'error_count': len([x for x in rename_log if x['status'] == 'error']),
            'success_rate': f"{len([x for x in rename_log if x['status'] == 'success']) / len(rename_log) * 100:.1f}%"
        }
        
        summary_file = self.output_dir.parent / 'rename_summary.txt'
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("=== 轴承数据片段重命名总结报告 ===\n\n")
            f.write(f"总文件数: {summary['total_files']}\n")
            f.write(f"成功重命名: {summary['success_count']}\n")
            f.write(f"失败数量: {summary['error_count']}\n")
            f.write(f"成功率: {summary['success_rate']}\n\n")
            
            f.write("=== 命名格式示例 ===\n")
            f.write("新格式: {采样频率}_{传感器位置}_data_{故障类型}_{故障尺寸}_{载荷}_{片段编号}\n")
            f.write("示例:\n")
            f.write("- 12kHz_DE_data_B_0007_0_1 (12kHz驱动端，滚动体故障，0.007英寸，0载荷，第1片段)\n")
            f.write("- 48kHz_DE_data_IR_0014_2_3 (48kHz驱动端，内圈故障，0.014英寸，2载荷，第3片段)\n")
            f.write("- 48kHz_Normal_data_N_0_2 (48kHz正常数据，0载荷，第2片段)\n\n")
            
            if summary['error_count'] > 0:
                f.write("=== 失败案例 ===\n")
                for log in rename_log:
                    if log['status'] == 'error':
                        f.write(f"原名: {log['old_name']} -> 目标: {log['new_name']}, 错误: {log.get('error', 'Unknown')}\n")
        
        logger.info(f"重命名日志保存至: {log_file}")
        logger.info(f"重命名总结保存至: {summary_file}")

def main():
    """
    主函数
    """
    print("🔄 开始轴承数据片段重命名...")
    
    renamer = SegmentRenamer()
    
    # 显示映射关系样例
    print("\n📋 数据集映射关系样例:")
    sample_mappings = list(renamer.dataset_mapping.items())[:5]
    for original_file, info in sample_mappings:
        print(f"  {original_file} -> {info}")
    
    print(f"\n📊 统计信息:")
    print(f"  原始数据文件数: {len(renamer.dataset_mapping)}")
    print(f"  当前片段文件夹数: {len(renamer.current_segments)}")
    print(f"  输出目录: {renamer.output_dir}")
    
    # 自动执行（非交互式）
    print(f"\n🚀 自动开始重命名...")
    print(f"⚠️  注意：将在新目录创建重命名后的文件，不会修改原文件")
    
    # 执行重命名
    rename_log = renamer.rename_segments()
    
    print("\n✅ 重命名完成！")
    print(f"📁 新文件夹位置: {renamer.output_dir}")
    print(f"📋 查看详细日志: {renamer.output_dir.parent / 'rename_log.csv'}")
    print(f"📄 查看总结报告: {renamer.output_dir.parent / 'rename_summary.txt'}")

if __name__ == "__main__":
    main()
