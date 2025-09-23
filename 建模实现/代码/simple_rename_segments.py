#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
轴承数据片段简化重命名程序
将final_segments中的数据片段重新命名，使其符合原始数据集的命名规范

命名规范：{采样频率}_{传感器位置}_data_{故障类型}_{故障尺寸}_{载荷}_{片段编号}
"""

import os
import shutil
from pathlib import Path
import logging
import csv

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleSegmentRenamer:
    def __init__(self):
        self.source_segments_dir = Path("/Users/gsyoung/MBP_documents/code/Matlab/研究生数学建模/建模实现/处理后数据/final_segments")
        self.output_dir = Path("/Users/gsyoung/MBP_documents/code/Matlab/研究生数学建模/建模实现/处理后数据/renamed_segments")
        
        # 清空并创建输出目录
        if self.output_dir.exists():
            shutil.rmtree(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def parse_segment_name(self, segment_name):
        """
        解析片段名称并返回标准化信息
        """
        # 移除末尾的编号（最后的_数字）
        parts = segment_name.split('_')
        if len(parts) < 3:
            return None
        
        # 去掉最后的编号
        name_without_id = '_'.join(parts[:-1])
        segment_num = parts[-1]
        
        # 解析不同的数据类型
        result = {
            'segment_num': segment_num,
            'sampling_freq': '12kHz',  # 默认12kHz
            'sensor_type': 'DE',       # 默认DE
            'fault_type': 'Unknown',
            'fault_size': '',
            'load': '0',
            'original_name': name_without_id
        }
        
        # 1. 正常数据 (N_*)
        if name_without_id.startswith('N_'):
            result['fault_type'] = 'N'
            result['sensor_type'] = 'Normal'
            result['sampling_freq'] = '48kHz'
            
            if 'N_0' in name_without_id:
                result['load'] = '0'
            elif 'N_1' in name_without_id:
                result['load'] = '1'
            elif 'N_2' in name_without_id:
                result['load'] = '2'
            elif 'N_3' in name_without_id:
                result['load'] = '3'
            return result
        
        # 2. 滚动体故障 (B*)
        if name_without_id.startswith('B'):
            result['fault_type'] = 'B'
            
            # 提取故障尺寸和载荷
            if 'B007_' in name_without_id:
                result['fault_size'] = '0007'
                result['load'] = name_without_id.split('_')[1]
            elif 'B014_' in name_without_id:
                result['fault_size'] = '0014'
                result['load'] = name_without_id.split('_')[1]
            elif 'B021_' in name_without_id:
                result['fault_size'] = '0021'
                result['load'] = name_without_id.split('_')[1]
            elif 'B028_' in name_without_id:
                result['fault_size'] = '0028'
                result['load'] = name_without_id.split('_')[1]
            
            # 判断数据集类型（12kHz还是48kHz）
            # 12kHz 数据集包含 0028 尺寸，48kHz 数据集不包含
            if result['fault_size'] == '0028':
                result['sampling_freq'] = '12kHz'
                result['sensor_type'] = 'DE'
            else:
                # 假设其他都是12kHz_FE（根据实际片段数量推断）
                result['sampling_freq'] = '12kHz'
                result['sensor_type'] = 'FE'
                
            return result
        
        # 3. 内圈故障 (IR*)
        if name_without_id.startswith('IR'):
            result['fault_type'] = 'IR'
            
            # 提取故障尺寸和载荷
            if 'IR007_' in name_without_id:
                result['fault_size'] = '0007'
                result['load'] = name_without_id.split('_')[1]
            elif 'IR014_' in name_without_id:
                result['fault_size'] = '0014'
                result['load'] = name_without_id.split('_')[1]
            elif 'IR021_' in name_without_id:
                result['fault_size'] = '0021'
                result['load'] = name_without_id.split('_')[1]
            elif 'IR028_' in name_without_id:
                result['fault_size'] = '0028'
                result['load'] = name_without_id.split('_')[1]
            
            # 判断数据集类型
            if result['fault_size'] == '0028':
                result['sampling_freq'] = '12kHz'
                result['sensor_type'] = 'DE'
            else:
                result['sampling_freq'] = '12kHz'
                result['sensor_type'] = 'FE'
                
            return result
        
        # 4. 外圈故障 (OR*)
        if name_without_id.startswith('OR'):
            result['fault_type'] = 'OR'
            
            # 解析OR故障格式：OR007@3_1 -> OR + 尺寸@位置_载荷
            if '@' in name_without_id:
                fault_part, load = name_without_id.rsplit('_', 1)
                result['load'] = load
                
                size_pos = fault_part.replace('OR', '')  # 007@3
                if '@' in size_pos:
                    size, position = size_pos.split('@')
                    result['fault_size'] = size.zfill(4)  # 确保4位数字
                    
                    # 根据位置判断数据集类型
                    # @3, @6, @12 通常表示不同的外圈位置
                    # 假设 @3 是 12kHz_DE, @6 是 48kHz_DE, @12 是 48kHz_DE
                    if position == '3':
                        result['sampling_freq'] = '12kHz'
                        result['sensor_type'] = 'DE'
                    elif position == '6':
                        result['sampling_freq'] = '48kHz'
                        result['sensor_type'] = 'DE'
                    elif position == '12':
                        result['sampling_freq'] = '48kHz'
                        result['sensor_type'] = 'DE'
                    else:
                        result['sampling_freq'] = '12kHz'
                        result['sensor_type'] = 'DE'
            
            return result
        
        logger.warning(f"无法解析片段名称: {name_without_id}")
        return None
    
    def generate_new_name(self, parse_result):
        """
        根据解析结果生成新的标准命名
        """
        if not parse_result:
            return None
        
        parts = [
            parse_result['sampling_freq'],    # 12kHz, 48kHz
            parse_result['sensor_type'],      # DE, FE, Normal
            'data',                           # 固定
            parse_result['fault_type'],       # B, IR, OR, N
        ]
        
        # 添加故障尺寸（如果有）
        if parse_result['fault_size']:
            parts.append(parse_result['fault_size'])
            
        # 添加载荷
        parts.append(parse_result['load'])
        
        # 添加片段编号
        parts.append(parse_result['segment_num'])
        
        return '_'.join(parts)
    
    def rename_segments(self):
        """
        执行批量重命名
        """
        rename_log = []
        success_count = 0
        error_count = 0
        duplicate_count = 0
        
        # 跟踪重复名称
        name_counter = {}
        
        for segment_dir in self.source_segments_dir.iterdir():
            if not segment_dir.is_dir():
                continue
                
            segment_name = segment_dir.name
            
            # 解析片段名称
            parse_result = self.parse_segment_name(segment_name)
            if not parse_result:
                error_count += 1
                rename_log.append({
                    'old_name': segment_name,
                    'new_name': '',
                    'status': 'parse_error',
                    'error': '无法解析片段名称'
                })
                continue
            
            # 生成新名称
            new_name = self.generate_new_name(parse_result)
            if not new_name:
                error_count += 1
                rename_log.append({
                    'old_name': segment_name,
                    'new_name': '',
                    'status': 'generate_error',
                    'error': '无法生成新名称'
                })
                continue
            
            # 处理重复名称
            if new_name in name_counter:
                name_counter[new_name] += 1
                new_name_unique = f"{new_name}_duplicate_{name_counter[new_name]}"
                duplicate_count += 1
            else:
                name_counter[new_name] = 0
                new_name_unique = new_name
            
            # 执行重命名
            old_path = segment_dir
            new_path = self.output_dir / new_name_unique
            
            try:
                # 复制整个文件夹
                shutil.copytree(old_path, new_path)
                
                # 重命名文件夹内的文件
                self._rename_internal_files(new_path, segment_name, new_name_unique)
                
                rename_log.append({
                    'old_name': segment_name,
                    'new_name': new_name_unique,
                    'original_mapped': parse_result['original_name'],
                    'status': 'success',
                    'fault_info': f"{parse_result['fault_type']}_{parse_result['fault_size']}_{parse_result['load']}"
                })
                success_count += 1
                
                if success_count % 50 == 0:
                    logger.info(f"已处理 {success_count} 个文件夹")
                    
            except Exception as e:
                logger.error(f"重命名失败: {segment_name} -> {new_name_unique}, 错误: {e}")
                rename_log.append({
                    'old_name': segment_name,
                    'new_name': new_name_unique,
                    'status': 'copy_error',
                    'error': str(e)
                })
                error_count += 1
        
        # 保存重命名日志
        self._save_rename_log(rename_log)
        
        logger.info(f"重命名完成！成功: {success_count}, 失败: {error_count}, 重复处理: {duplicate_count}")
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
        保存重命名日志（使用CSV而不是pandas）
        """
        log_file = self.output_dir.parent / 'simple_rename_log.csv'
        
        # 写入CSV文件
        with open(log_file, 'w', newline='', encoding='utf-8') as csvfile:
            if rename_log:
                fieldnames = rename_log[0].keys()
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rename_log)
        
        # 生成总结报告
        summary = {
            'total_files': len(rename_log),
            'success_count': len([x for x in rename_log if x['status'] == 'success']),
            'error_count': len([x for x in rename_log if x['status'] != 'success']),
        }
        summary['success_rate'] = f"{summary['success_count'] / summary['total_files'] * 100:.1f}%" if summary['total_files'] > 0 else "0%"
        
        summary_file = self.output_dir.parent / 'simple_rename_summary.txt'
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("=== 轴承数据片段简化重命名总结报告 ===\n\n")
            f.write(f"总文件数: {summary['total_files']}\n")
            f.write(f"成功重命名: {summary['success_count']}\n")
            f.write(f"失败数量: {summary['error_count']}\n")
            f.write(f"成功率: {summary['success_rate']}\n\n")
            
            f.write("=== 新命名格式示例 ===\n")
            f.write("格式: {采样频率}_{传感器位置}_data_{故障类型}_{故障尺寸}_{载荷}_{片段编号}\n")
            f.write("示例:\n")
            f.write("- 12kHz_DE_data_B_0007_0_1 (12kHz驱动端，滚动体故障，0.007英寸，0载荷，第1片段)\n")
            f.write("- 12kHz_FE_data_IR_0014_2_3 (12kHz风扇端，内圈故障，0.014英寸，2载荷，第3片段)\n")
            f.write("- 48kHz_Normal_data_N_0_2 (48kHz正常数据，0载荷，第2片段)\n\n")
            
            if summary['error_count'] > 0:
                f.write("=== 失败案例（前10个） ===\n")
                error_cases = [log for log in rename_log if log['status'] != 'success'][:10]
                for log in error_cases:
                    f.write(f"原名: {log['old_name']} -> 状态: {log['status']}, 错误: {log.get('error', 'Unknown')}\n")
        
        logger.info(f"重命名日志保存至: {log_file}")
        logger.info(f"重命名总结保存至: {summary_file}")

def main():
    """
    主函数
    """
    print("🔄 开始轴承数据片段简化重命名...")
    
    renamer = SimpleSegmentRenamer()
    
    print(f"\n📊 基本信息:")
    segments = list(renamer.source_segments_dir.iterdir())
    print(f"  当前片段文件夹数: {len([s for s in segments if s.is_dir()])}")
    print(f"  输出目录: {renamer.output_dir}")
    
    print(f"\n🚀 开始重命名...")
    print(f"⚠️  注意：将在新目录创建重命名后的文件，不会修改原文件")
    
    # 执行重命名
    rename_log = renamer.rename_segments()
    
    print("\n✅ 重命名完成！")
    print(f"📁 新文件夹位置: {renamer.output_dir}")
    print(f"📋 查看详细日志: {renamer.output_dir.parent / 'simple_rename_log.csv'}")
    print(f"📄 查看总结报告: {renamer.output_dir.parent / 'simple_rename_summary.txt'}")

if __name__ == "__main__":
    main()
