#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import fft
import shutil

# 源数据路径和目标路径
src_dir = '/home/gsyoung/Documents/Code/研究生竞赛/数学建模/2025E/2025_E/建模实现/处理后数据/raw_data'
dst_dir = '/home/gsyoung/Documents/Code/研究生竞赛/数学建模/2025E/2025_E/建模实现/处理后数据/切片数据'

# 确保目标目录存在
os.makedirs(dst_dir, exist_ok=True)

# 获取所有子文件夹
subdirs = [d for d in os.listdir(src_dir) if os.path.isdir(os.path.join(src_dir, d))]
print(f"找到 {len(subdirs)} 个子文件夹")

# 处理每个子文件夹
for subdir in subdirs:
    src_subdir = os.path.join(src_dir, subdir)
    dst_subdir = os.path.join(dst_dir, subdir)
    
    # 创建目标子文件夹
    os.makedirs(dst_subdir, exist_ok=True)
    
    # 查找npy文件
    npy_files = [f for f in os.listdir(src_subdir) if f.endswith('_raw_data.npy')]
    
    if not npy_files:
        print(f"警告: 在 {subdir} 中未找到npy文件")
        continue
    
    npy_file = npy_files[0]  # 假设每个文件夹只有一个npy文件
    npy_path = os.path.join(src_subdir, npy_file)
    
    try:
        # 读取原始数据
        raw_data = np.load(npy_path)
        
        # 从中间部分截取3000个点
        total_points = raw_data.shape[0]
        mid_point = total_points // 2
        start_idx = mid_point - 1500
        end_idx = mid_point + 1500
        
        # 确保索引在有效范围内
        if start_idx < 0:
            start_idx = 0
            end_idx = 3000
        if end_idx > total_points:
            end_idx = total_points
            start_idx = total_points - 3000 if total_points > 3000 else 0
        
        # 截取数据
        sliced_data = raw_data[start_idx:end_idx]
        
        # 如果截取的数据点数不足3000，则进行处理
        if len(sliced_data) < 3000:
            print(f"警告: {subdir} 中的数据点数不足3000，实际为 {len(sliced_data)}")
            if len(sliced_data) == 0:
                print(f"错误: {subdir} 中无有效数据，跳过处理")
                continue
        
        # 保存截取的数据
        sliced_npy_path = os.path.join(dst_subdir, f"{subdir}_sliced_data.npy")
        np.save(sliced_npy_path, sliced_data)
        
        # 计算频域数据
        N = len(sliced_data)
        yf = fft.fft(sliced_data)
        xf = fft.fftfreq(N, 1/48000)[:N//2]  # 假设采样率为48kHz
        yf_abs = 2.0/N * np.abs(yf[:N//2])
        
        # 保存频域数据
        freq_data = np.column_stack((xf, yf_abs))
        freq_csv_path = os.path.join(dst_subdir, f"{subdir}_frequency_analysis.csv")
        np.savetxt(freq_csv_path, freq_data, delimiter=',', header='Frequency,Amplitude', comments='')
        
        # 绘制时域图
        plt.figure(figsize=(10, 4))
        plt.plot(np.arange(len(sliced_data))/48000, sliced_data)
        plt.title(f'{subdir} - Time Domain')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.grid(True)
        time_domain_path = os.path.join(dst_subdir, f"{subdir}_time_domain.png")
        plt.savefig(time_domain_path, dpi=300)
        plt.close()
        
        # 绘制频域图
        plt.figure(figsize=(10, 4))
        plt.plot(xf, yf_abs)
        plt.title(f'{subdir} - Frequency Domain')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Amplitude')
        plt.grid(True)
        plt.xlim(0, 10000)  # 限制频率范围以便更好地查看
        freq_domain_path = os.path.join(dst_subdir, f"{subdir}_frequency_domain.png")
        plt.savefig(freq_domain_path, dpi=300)
        plt.close()
        
        # 创建特征文件
        features = {
            'mean': np.mean(sliced_data),
            'std': np.std(sliced_data),
            'max': np.max(sliced_data),
            'min': np.min(sliced_data),
            'rms': np.sqrt(np.mean(sliced_data**2)),
            'peak_freq': xf[np.argmax(yf_abs)]
        }
        
        features_csv_path = os.path.join(dst_subdir, f"{subdir}_features.csv")
        with open(features_csv_path, 'w') as f:
            f.write("Feature,Value\n")
            for key, value in features.items():
                f.write(f"{key},{value}\n")
        
        print(f"已处理 {subdir}")
    
    except Exception as e:
        print(f"处理 {subdir} 时出错: {e}")

print("所有数据处理完成")
