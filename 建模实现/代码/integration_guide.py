#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
集成新去噪方法到现有 raw_data_processor.py 的指导代码
"""

def upgrade_raw_data_processor():
    """
    展示如何将增强去噪方法集成到现有的 raw_data_processor.py 中
    """
    
    print("🔧 集成新去噪方法到现有流程")
    print("=" * 50)
    
    print("1️⃣ 修改导入部分:")
    print("""
# 在 raw_data_processor.py 文件顶部添加：
from enhanced_denoising_methods import EnhancedDenoising
    """)
    
    print("2️⃣ 修改 __init__ 方法:")
    print("""
def __init__(self):
    # ... 现有代码 ...
    
    # 添加增强去噪器
    self.enhanced_denoiser = EnhancedDenoising()
    self.use_enhanced_denoising = True  # 控制是否使用新方法
    """)
    
    print("3️⃣ 升级 apply_denoising 方法:")
    print("""
def apply_denoising(self, data, fs):
    '''应用去噪滤波器'''
    if self.use_enhanced_denoising:
        # 使用新的增强去噪方法
        try:
            return self.enhanced_denoiser.auto_denoising(data, fs)
        except Exception as e:
            print(f"⚠️ 增强去噪失败，使用传统方法: {e}")
            return self.apply_denoising_traditional(data, fs)
    else:
        return self.apply_denoising_traditional(data, fs)

def apply_denoising_traditional(self, data, fs):
    '''传统去噪方法（备份）'''
    # ... 您现有的去噪代码 ...
    """)
    
    print("4️⃣ 添加方法选择功能:")
    print("""
def set_denoising_method(self, method='auto'):
    '''设置去噪方法'''
    if method in ['auto', 'wavelet', 'emd', 'vmd', 'enhanced']:
        self.use_enhanced_denoising = True
        self.enhanced_denoiser.preferred_method = method
        print(f"✅ 已切换到增强去噪方法: {method}")
    elif method == 'traditional':
        self.use_enhanced_denoising = False
        print("✅ 已切换到传统去噪方法")
    else:
        print(f"⚠️ 未知方法: {method}")
    """)
    
    print("5️⃣ 使用示例:")
    print("""
# 在处理数据时：
processor = RawDataProcessor()

# 选择去噪方法
processor.set_denoising_method('auto')  # 智能自动选择
# processor.set_denoising_method('wavelet')  # 强制使用小波
# processor.set_denoising_method('traditional')  # 使用原方法

# 正常处理数据
processor.process_all_files()
    """)

def create_enhanced_raw_data_processor():
    """
    创建一个完整的增强版 raw_data_processor.py 示例
    """
    from pathlib import Path
    enhanced_code = '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强版原始数据处理器
集成了先进的去噪方法
"""

import numpy as np
import scipy.io
import scipy.signal as signal
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 导入增强去噪方法
try:
    from enhanced_denoising_methods import EnhancedDenoising
    ENHANCED_DENOISING_AVAILABLE = True
    print("✅ 增强去噪方法可用")
except ImportError:
    ENHANCED_DENOISING_AVAILABLE = False
    print("⚠️ 增强去噪方法不可用，将使用传统方法")


class EnhancedRawDataProcessor:
    """增强版原始数据处理器"""
    
    def __init__(self):
        self.output_base_path = Path("/Users/gsyoung/MBP_documents/code/Matlab/研究生数学建模/建模实现/处理后数据/raw_data")
        self.bearing_params = {
            'SKF6205': {'n': 9, 'd': 0.3126, 'D': 1.537},
            'SKF6203': {'n': 9, 'd': 0.2656, 'D': 1.122}
        }
        
        # 初始化增强去噪器
        if ENHANCED_DENOISING_AVAILABLE:
            self.enhanced_denoiser = EnhancedDenoising()
            self.use_enhanced_denoising = True
            self.denoising_method = 'auto'  # auto, wavelet, emd, vmd, enhanced, traditional
        else:
            self.use_enhanced_denoising = False
            self.denoising_method = 'traditional'
    
    def set_denoising_method(self, method='auto'):
        """设置去噪方法"""
        if method in ['auto', 'wavelet', 'emd', 'vmd', 'enhanced'] and ENHANCED_DENOISING_AVAILABLE:
            self.use_enhanced_denoising = True
            self.denoising_method = method
            print(f"✅ 已设置去噪方法: {method}")
        elif method == 'traditional':
            self.use_enhanced_denoising = False
            self.denoising_method = 'traditional'
            print("✅ 已设置为传统去噪方法")
        else:
            print(f"⚠️ 方法 '{method}' 不可用，使用传统方法")
            self.use_enhanced_denoising = False
            self.denoising_method = 'traditional'
    
    def apply_denoising(self, data, fs):
        """应用去噪滤波器（增强版）"""
        if self.use_enhanced_denoising and ENHANCED_DENOISING_AVAILABLE:
            try:
                if self.denoising_method == 'auto':
                    return self.enhanced_denoiser.auto_denoising(data, fs)
                elif self.denoising_method == 'wavelet':
                    return self.enhanced_denoiser.wavelet_denoising(data)
                elif self.denoising_method == 'emd':
                    return self.enhanced_denoiser.emd_denoising(data)
                elif self.denoising_method == 'vmd':
                    return self.enhanced_denoiser.vmd_denoising(data)
                elif self.denoising_method == 'enhanced':
                    return self.enhanced_denoiser.enhanced_traditional_denoising(data, fs)
                else:
                    return self.enhanced_denoiser.auto_denoising(data, fs)
            except Exception as e:
                print(f"⚠️ 增强去噪失败: {e}，使用传统方法")
                return self.apply_denoising_traditional(data, fs)
        else:
            return self.apply_denoising_traditional(data, fs)
    
    def apply_denoising_traditional(self, data, fs):
        """传统去噪方法（备份）"""
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
    
    # ... 其他方法保持不变 ...


def main():
    """演示增强版处理器的使用"""
    print("🚀 增强版原始数据处理器演示")
    print("=" * 50)
    
    processor = EnhancedRawDataProcessor()
    
    # 测试不同去噪方法
    test_methods = ['traditional', 'auto', 'wavelet']
    
    for method in test_methods:
        print(f"\\n🧪 测试去噪方法: {method}")
        processor.set_denoising_method(method)
        
        # 这里可以继续处理数据...
        print(f"   当前设置: {'增强' if processor.use_enhanced_denoising else '传统'}")


if __name__ == "__main__":
    main()
'''
    
    # 保存增强版代码
    output_path = Path("/Users/gsyoung/MBP_documents/code/Matlab/研究生数学建模/建模实现/代码/enhanced_raw_data_processor.py")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(enhanced_code)
    
    print(f"📁 增强版处理器已保存到: {output_path}")
    return output_path

if __name__ == "__main__":
    upgrade_raw_data_processor()
    print("\\n" + "="*50)
    enhanced_file = create_enhanced_raw_data_processor()
    print(f"\\n✅ 集成指导完成！")
    print(f"📁 增强版代码: {enhanced_file}")
