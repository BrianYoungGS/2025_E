# 🔧 轴承故障诊断模型训练模块

## 📋 **模块概览**

这是一个完整的轴承故障诊断模型训练框架，专门用于处理轴承振动信号数据，支持四分类任务（N、B、IR、OR），并支持迁移学习到高速列车应用。

### 🎯 **核心功能**
- **多模态数据加载**: 支持原始信号、特征数据、频域分析、图像数据
- **智能数据筛选**: 按故障类型、采样频率、传感器位置等条件筛选
- **完整预处理**: 标准化、特征工程、数据增强、数据平衡
- **灵活配置**: 预定义的实验配置，支持快速实验迭代
- **迁移学习**: 支持不同域之间的知识迁移

---

## 📁 **文件结构**

```
model/
├── data_loader.py              # 🔌 核心数据加载器
├── data_preprocessor.py        # 🔧 数据预处理模块
├── config.py                   # ⚙️ 配置管理
├── example_data_loading.py     # 📖 完整使用示例
├── test_data_loader.py         # 🧪 简化测试脚本
└── README.md                   # 📚 本文档
```

---

## 🚀 **快速开始**

### **第一步：基础数据加载**

```python
from data_loader import BearingDataLoader

# 创建数据加载器
loader = BearingDataLoader()

# 扫描所有可用数据
segment_info, raw_info = loader.scan_datasets()

# 获取数据集统计
stats = loader.get_dataset_statistics()
print("数据集统计:", stats)
```

### **第二步：筛选训练数据**

```python
# 筛选12kHz驱动端数据用于训练
train_data = loader.filter_by_criteria(
    fault_types=['N', 'B', 'IR', 'OR'],  # 四种故障类型
    sampling_freqs=['12kHz'],             # 12kHz数据
    sensor_types=['DE'],                  # 驱动端传感器
    data_type='segment',                  # 分段数据
    loads=[0, 1, 2, 3]                   # 所有载荷条件
)
```

### **第三步：加载多模态数据**

```python
# 加载完整数据集
dataset = loader.load_dataset(
    train_data,
    include_raw=True,           # 原始信号数据
    include_features=True,      # 特征数据
    include_freq_analysis=True, # 频域分析数据
    include_images=False        # 图像数据（可选）
)

print(f"原始信号: {dataset['raw_signals'].shape}")
print(f"特征数据: {dataset['features'].shape}")
print(f"标签: {dataset['labels'].shape}")
```

---

## 📊 **数据集详细信息**

### **🔍 数据来源**
- **`@renamed_segments/`**: 483个分段数据文件夹
- **`@raw_data/`**: 322个完整原始数据文件夹（161个原始 + 161个去噪）

### **📋 数据分布统计**
根据测试结果：
- **总样本数**: 785个
- **分段数据**: 483个
- **原始数据**: 302个
- **故障分布**: N(16), B(192), IR(192), OR(385)
- **频率分布**: 12kHz(665), 48kHz(120)
- **传感器分布**: DE(463), FE(306), Normal(16)

### **🏷️ 故障类型说明**
- **N (Normal)**: 正常状态 - 标签 0
- **B (Ball)**: 滚动体故障 - 标签 1  
- **IR (Inner Race)**: 内圈故障 - 标签 2
- **OR (Outer Race)**: 外圈故障 - 标签 3

### **📁 文件内容结构**
每个数据文件夹包含5个文件：
```
{数据名称}/
├── {数据名称}_features.csv          # 时域和频域特征
├── {数据名称}_frequency_analysis.csv # 频域分析数据
├── {数据名称}_frequency_domain.png   # 频域图像
├── {数据名称}_raw_data.npy           # 原始时间序列
└── {数据名称}_time_domain.png        # 时域图像
```

---

## 🔧 **数据预处理**

### **信号标准化**

```python
from data_preprocessor import BearingDataPreprocessor

preprocessor = BearingDataPreprocessor()

# 标准化原始信号
normalized_signals = preprocessor.normalize_signals(
    dataset['raw_signals'], 
    method='standard'  # 'standard', 'minmax', 'robust'
)
```

### **特征工程**

```python
# 提取综合特征（时域+频域+小波）
extracted_features = preprocessor.extract_comprehensive_features(
    normalized_signals,
    fs=12000,                    # 采样频率
    include_wavelet=True         # 包含小波特征
)

# 特征选择
selected_features = preprocessor.feature_selection(
    extracted_features,
    dataset['labels'],
    method='univariate',         # 'univariate' 或 'pca'
    k=50                        # 选择50个最佳特征
)
```

### **数据增强**

```python
# 数据增强（用于训练集）
augmented_signals, augmented_labels = preprocessor.augment_signals(
    normalized_signals,
    dataset['labels'],
    methods=['noise', 'scaling'], # 噪声+幅值缩放
    noise_level=0.01
)
```

---

## ⚙️ **配置管理**

### **使用预定义配置**

```python
from config import get_experiment_config

# 获取实验配置
config = get_experiment_config('experiment_1')
print("实验配置:", config['experiment']['name'])
print("数据集:", config['dataset']['name'])
print("模型类型:", config['model']['model_type'])
```

### **可用的实验配置**
- **`experiment_1`**: 基础分类实验（12kHz DE + Random Forest）
- **`experiment_2`**: 深度学习分类（12kHz DE + CNN）
- **`experiment_3`**: 传感器迁移（DE→FE + CNN + Fine-tuning）
- **`experiment_4`**: 频率迁移（12kHz→48kHz + Transformer）
- **`experiment_5`**: 混合域实验（多域数据 + XGBoost）

---

## 🎯 **迁移学习应用**

### **场景1：传感器位置迁移**
```python
# 源域：12kHz驱动端数据
source_data = loader.filter_by_criteria(
    sampling_freqs=['12kHz'],
    sensor_types=['DE'],
    data_type='segment'
)

# 目标域：12kHz风扇端数据
target_data = loader.filter_by_criteria(
    sampling_freqs=['12kHz'],
    sensor_types=['FE'],
    data_type='segment'
)
```

### **场景2：采样频率迁移**
```python
# 源域：12kHz数据
source_data = loader.filter_by_criteria(
    sampling_freqs=['12kHz'],
    data_type='segment'
)

# 目标域：48kHz数据
target_data = loader.filter_by_criteria(
    sampling_freqs=['48kHz'],
    data_type='raw'
)
```

---

## 📈 **实际应用示例**

### **示例1：基础四分类模型**

```python
# 完整的训练流程
from data_loader import BearingDataLoader
from data_preprocessor import create_preprocessing_pipeline

# 1. 数据加载
loader = BearingDataLoader()
train_data = loader.filter_by_criteria(
    fault_types=['N', 'B', 'IR', 'OR'],
    sampling_freqs=['12kHz'],
    sensor_types=['DE'],
    data_type='segment'
)

dataset = loader.load_dataset(train_data, include_raw=True)

# 2. 数据预处理
processed_signals, processed_labels = create_preprocessing_pipeline(
    dataset['raw_signals'],
    dataset['labels'],
    config={
        'normalize': True,
        'augment': True,
        'balance': True
    }
)

# 3. 准备用于模型训练
print(f"预处理后数据: {processed_signals.shape}")
print(f"标签分布: {np.bincount(processed_labels)}")
```

### **示例2：特征工程流程**

```python
from data_preprocessor import BearingDataPreprocessor

preprocessor = BearingDataPreprocessor()

# 提取多种特征
time_features = preprocessor.extract_time_domain_features(signals)
freq_features = preprocessor.extract_frequency_domain_features(signals, fs=12000)
wavelet_features = preprocessor.extract_wavelet_features(signals)

# 合并所有特征
all_features = np.concatenate([time_features, freq_features, wavelet_features], axis=1)

# 特征选择和标准化
selected_features = preprocessor.feature_selection(all_features, labels, k=100)
normalized_features = preprocessor.normalize_signals(selected_features, method='standard')
```

---

## 📊 **性能基准**

根据数据集测试结果：

### **数据量统计**
- ✅ **分段数据**: 483个样本（每个约20k点）
- ✅ **原始数据**: 302个样本（每个约60k-400k点）
- ✅ **文件完整性**: 100%（所有样本都包含5个必需文件）

### **数据质量**
- ✅ **命名规范**: 100%符合标准格式
- ✅ **解析成功率**: >95%（少数带转速信息的文件需要额外处理）
- ✅ **故障分布**: 涵盖所有四种故障类型
- ✅ **多域支持**: 支持不同频率、传感器、载荷条件

---

## 🚀 **后续模型训练步骤**

### **第一阶段：传统机器学习**
1. 使用提取的特征训练Random Forest/SVM/XGBoost
2. 验证四分类性能
3. 建立性能基准

### **第二阶段：深度学习**
1. 使用原始信号训练CNN/LSTM
2. 对比特征工程vs端到端学习
3. 优化网络结构

### **第三阶段：迁移学习**
1. 在源域训练基础模型
2. 迁移到不同目标域
3. 验证泛化能力

### **第四阶段：高速列车应用**
1. 域自适应技术
2. 增量学习
3. 实时故障检测

---

## 💡 **使用建议**

### **数据选择策略**
- **原型验证**: 使用分段数据快速验证算法
- **深度学习**: 使用原始数据训练端到端模型
- **迁移学习**: 混合使用不同域的数据

### **预处理策略**
- **传统ML**: 重点关注特征工程
- **深度学习**: 重点关注数据增强和标准化
- **迁移学习**: 重点关注域对齐

### **实验流程**
1. 先用小样本快速验证思路
2. 逐步扩大数据规模
3. 系统化对比不同方法
4. 最终在完整数据集上验证

---

## 🔧 **故障排除**

### **常见问题**

**Q1: 导入模块失败**
```bash
# 安装必需的依赖包
pip install numpy pandas scikit-learn opencv-python
pip install PyWavelets  # 用于小波特征（可选）
```

**Q2: 内存不足**
```python
# 分批加载数据
dataset = loader.load_dataset(train_data[:100])  # 先加载100个样本
```

**Q3: 文件路径错误**
```python
# 检查数据目录
loader = BearingDataLoader("/your/custom/path")
```

---

## 📞 **技术支持**

- **测试脚本**: 运行 `python test_data_loader.py` 验证环境
- **完整示例**: 参考 `example_data_loading.py`
- **配置参考**: 查看 `config.py` 中的预定义配置

---

**🎉 现在您已经拥有了完整的轴承故障诊断数据处理框架！开始您的模型训练之旅吧！**
