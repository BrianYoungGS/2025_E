#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
改进版MHDCNN模型
Improved Multi-scale Hybrid Dilated CNN for Bearing Fault Diagnosis

主要改进：
1. 更稳定的训练策略
2. 改进的数据预处理
3. 更好的正则化
4. 自适应学习率调度
5. 数据平衡处理
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import numpy as np
import pandas as pd
import cv2
import os
import json
import logging
from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImprovedResidualBlock(nn.Module):
    """改进的残差块，加入更好的正则化"""
    def __init__(self, in_channels, out_channels, stride=1, dropout=0.1):
        super(ImprovedResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.dropout1 = nn.Dropout1d(dropout)
        
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout2 = nn.Dropout1d(dropout)
        
        self.se = SEBlock(out_channels)  # 添加SE注意力机制
        
        # 维度匹配
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )
    
    def forward(self, x):
        residual = self.shortcut(x)
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout1(out)
        
        out = self.bn2(self.conv2(out))
        out = self.dropout2(out)
        
        # SE注意力
        out = self.se(out)
        
        out += residual
        out = F.relu(out)
        
        return out

class SEBlock(nn.Module):
    """SE注意力机制块"""
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)

class ImprovedMultiScaleDilatedConv2D(nn.Module):
    """改进的多尺度混合空洞卷积"""
    def __init__(self, in_channels, out_channels, dropout=0.1):
        super(ImprovedMultiScaleDilatedConv2D, self).__init__()
        
        # 多尺度分支
        self.conv_1 = nn.Conv2d(in_channels, out_channels//4, kernel_size=3, 
                               padding=1, dilation=1)
        self.conv_2 = nn.Conv2d(in_channels, out_channels//4, kernel_size=3, 
                               padding=2, dilation=2)
        self.conv_3 = nn.Conv2d(in_channels, out_channels//4, kernel_size=3, 
                               padding=4, dilation=4)
        self.conv_4 = nn.Conv2d(in_channels, out_channels//4, kernel_size=3, 
                               padding=8, dilation=8)
        
        # 批标准化和Dropout
        self.bn = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(dropout)
        
        # 通道注意力
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, out_channels//8, 1),
            nn.ReLU(),
            nn.Conv2d(out_channels//8, out_channels, 1),
            nn.Sigmoid()
        )
        
        # 特征融合
        self.fusion_conv = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        
    def forward(self, x):
        # 多尺度特征
        out1 = self.conv_1(x)
        out2 = self.conv_2(x)
        out3 = self.conv_3(x)
        out4 = self.conv_4(x)
        
        # 拼接
        out = torch.cat([out1, out2, out3, out4], dim=1)
        
        # 标准化和dropout
        out = F.relu(self.bn(out))
        out = self.dropout(out)
        
        # 通道注意力
        att = self.channel_attention(out)
        out = out * att
        
        # 特征融合
        out = self.fusion_conv(out)
        
        return out

class ImprovedImageEncoder(nn.Module):
    """改进的图像编码器"""
    def __init__(self, input_channels=3, feature_dim=256):
        super(ImprovedImageEncoder, self).__init__()
        
        # 多尺度混合空洞卷积层
        self.mhdc1 = ImprovedMultiScaleDilatedConv2D(input_channels, 64, dropout=0.1)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.mhdc2 = ImprovedMultiScaleDilatedConv2D(64, 128, dropout=0.1)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.mhdc3 = ImprovedMultiScaleDilatedConv2D(128, 256, dropout=0.15)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        self.mhdc4 = ImprovedMultiScaleDilatedConv2D(256, 512, dropout=0.15)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        # 改进的全连接层
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512 * 4 * 4, feature_dim * 2),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(feature_dim * 2),
            nn.Dropout(0.3),
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        x = self.mhdc1(x)
        x = self.pool1(x)
        
        x = self.mhdc2(x)
        x = self.pool2(x)
        
        x = self.mhdc3(x)
        x = self.pool3(x)
        
        x = self.mhdc4(x)
        x = self.adaptive_pool(x)
        
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        
        return x

class ImprovedSequenceEncoder(nn.Module):
    """改进的序列编码器"""
    def __init__(self, input_length=20000, feature_dim=256):
        super(ImprovedSequenceEncoder, self).__init__()
        
        # 初始卷积
        self.conv1 = nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm1d(64)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        # 改进的残差层
        self.layer1 = self._make_layer(64, 64, 2, dropout=0.1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2, dropout=0.1)
        self.layer3 = self._make_layer(128, 256, 3, stride=2, dropout=0.15)  # 增加层数
        self.layer4 = self._make_layer(256, 512, 2, stride=2, dropout=0.15)
        
        # 全局平均池化和最大池化的组合
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        
        # 改进的全连接
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512 * 2, feature_dim * 2),  # *2因为avg和max池化拼接
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(feature_dim * 2),
            nn.Dropout(0.3),
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(inplace=True)
        )
        
    def _make_layer(self, in_channels, out_channels, blocks, stride=1, dropout=0.1):
        layers = []
        layers.append(ImprovedResidualBlock(in_channels, out_channels, stride, dropout))
        for _ in range(1, blocks):
            layers.append(ImprovedResidualBlock(out_channels, out_channels, dropout=dropout))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # 组合池化
        avg_out = self.avg_pool(x).view(x.size(0), -1)
        max_out = self.max_pool(x).view(x.size(0), -1)
        x = torch.cat([avg_out, max_out], dim=1)
        
        x = self.fc(x)
        return x

class ImprovedTransformerEncoder(nn.Module):
    """改进的Transformer编码器"""
    def __init__(self, input_dim, feature_dim=256, nhead=8, num_layers=4, dropout=0.1):
        super(ImprovedTransformerEncoder, self).__init__()
        
        # 输入投影
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.Dropout(dropout)
        )
        
        # 位置编码
        self.pos_encoding = nn.Parameter(torch.randn(1000, feature_dim) * 0.1)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=nhead,
            dim_feedforward=feature_dim * 4,
            dropout=dropout,
            activation='gelu',  # 使用GELU激活
            batch_first=True,
            norm_first=True  # Pre-norm结构
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 输出投影
        self.output_projection = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # 投影
        x = self.input_projection(x)
        
        # 位置编码
        x = x.unsqueeze(1)  # (batch_size, 1, feature_dim)
        x = x + self.pos_encoding[:1, :].unsqueeze(0)
        
        # Transformer编码
        x = self.transformer(x)
        
        # 输出
        x = x.squeeze(1)
        x = self.output_projection(x)
        
        return x

class ImprovedMultiModalFusionClassifier(nn.Module):
    """改进的多模态融合分类器"""
    def __init__(self, feature_dim=256, num_classes=4, dropout=0.3):
        super(ImprovedMultiModalFusionClassifier, self).__init__()
        
        # 多头注意力融合
        self.attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # 模态权重学习
        self.modal_weights = nn.Parameter(torch.ones(3) / 3)
        
        # 改进的融合网络
        self.fusion = nn.Sequential(
            nn.Linear(feature_dim * 3, feature_dim * 2),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(feature_dim * 2),
            nn.Dropout(dropout),
            
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(feature_dim),
            nn.Dropout(dropout * 0.5)
        )
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(feature_dim // 2),
            nn.Dropout(dropout * 0.5),
            
            nn.Linear(feature_dim // 2, feature_dim // 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.3),
            
            nn.Linear(feature_dim // 4, num_classes)
        )
        
    def forward(self, image_features, sequence_features, csv_features):
        # 模态权重softmax
        weights = F.softmax(self.modal_weights, dim=0)
        
        # 加权特征
        weighted_features = torch.stack([
            image_features * weights[0],
            sequence_features * weights[1], 
            csv_features * weights[2]
        ], dim=1)
        
        # 注意力融合
        attended_features, _ = self.attention(weighted_features, weighted_features, weighted_features)
        
        # 展平并融合
        attended_features = attended_features.reshape(attended_features.size(0), -1)
        fused_features = self.fusion(attended_features)
        
        # 分类
        output = self.classifier(fused_features)
        
        return output

class ImprovedMHDCNN(nn.Module):
    """改进版MHDCNN模型"""
    def __init__(self, csv_input_dim, sequence_length=20000, feature_dim=256, num_classes=4):
        super(ImprovedMHDCNN, self).__init__()
        
        # 三个编码器
        self.image_encoder = ImprovedImageEncoder(input_channels=3, feature_dim=feature_dim)
        self.sequence_encoder = ImprovedSequenceEncoder(input_length=sequence_length, feature_dim=feature_dim)
        self.csv_encoder = ImprovedTransformerEncoder(input_dim=csv_input_dim, feature_dim=feature_dim)
        
        # 融合分类器
        self.fusion_classifier = ImprovedMultiModalFusionClassifier(feature_dim=feature_dim, num_classes=num_classes)
        
        # 权重初始化
        self._initialize_weights()
        
    def _initialize_weights(self):
        """改进的权重初始化"""
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, images, sequences, csv_features):
        # 三个分支编码
        image_features = self.image_encoder(images)
        sequence_features = self.sequence_encoder(sequences)
        csv_features_encoded = self.csv_encoder(csv_features)
        
        # 融合分类
        output = self.fusion_classifier(image_features, sequence_features, csv_features_encoded)
        
        return output

class ImprovedBearingDataset(Dataset):
    """改进的轴承数据集类"""
    def __init__(self, data_dir, fold_num=None, is_training=True, use_augmentation=True):
        self.data_dir = Path(data_dir)
        self.is_training = is_training
        self.use_augmentation = use_augmentation and is_training
        
        # 故障类型映射
        self.fault_mapping = {'N': 0, 'B': 1, 'IR': 2, 'OR': 3}
        
        # 加载数据路径
        if fold_num is not None:
            self.fold_dir = self.data_dir / f"fold_{fold_num:02d}"
            self.data_paths = list(self.fold_dir.iterdir())
        else:
            self.data_paths = []
            for fold_dir in self.data_dir.glob("fold_*"):
                self.data_paths.extend(list(fold_dir.iterdir()))
        
        self.data_paths = [p for p in self.data_paths if p.is_dir()]
        
        # 改进的特征标准化
        self.scaler = RobustScaler()  # 使用RobustScaler，对异常值更鲁棒
        if is_training:
            self._fit_scaler()
        
        logger.info(f"加载数据集: {len(self.data_paths)} 个样本")
    
    def _fit_scaler(self):
        """拟合特征标准化器"""
        all_features = []
        max_dim = 0
        
        # 确定最大特征维度
        for data_path in self.data_paths[:200]:  # 使用更多样本拟合
            try:
                features_file = data_path / f"{data_path.name}_features.csv"
                freq_file = data_path / f"{data_path.name}_frequency_analysis.csv"
                
                csv_features = []
                if features_file.exists():
                    features_df = pd.read_csv(features_file)
                    numeric_features = features_df.select_dtypes(include=[np.number]).values.flatten()
                    csv_features.extend(numeric_features)
                
                if freq_file.exists():
                    freq_df = pd.read_csv(freq_file)
                    numeric_freq = freq_df.select_dtypes(include=[np.number]).values.flatten()
                    csv_features.extend(numeric_freq)
                
                if csv_features:
                    max_dim = max(max_dim, len(csv_features))
            except Exception as e:
                continue
        
        self.csv_feature_dim = max(max_dim, 64)  # 增加最小维度
        
        # 收集特征
        for data_path in self.data_paths[:200]:
            try:
                csv_features = self._load_csv_features(data_path)
                if csv_features is not None:
                    all_features.append(csv_features)
            except Exception as e:
                all_features.append(np.zeros(self.csv_feature_dim, dtype=np.float32))
                continue
        
        if all_features:
            self.scaler.fit(np.array(all_features))
    
    def _load_csv_features(self, data_path):
        """加载CSV特征的辅助函数"""
        features_file = data_path / f"{data_path.name}_features.csv"
        freq_file = data_path / f"{data_path.name}_frequency_analysis.csv"
        
        csv_features = []
        if features_file.exists():
            try:
                features_df = pd.read_csv(features_file)
                numeric_features = features_df.select_dtypes(include=[np.number]).values.flatten()
                csv_features.extend(numeric_features)
            except Exception as e:
                pass
        
        if freq_file.exists():
            try:
                freq_df = pd.read_csv(freq_file)
                numeric_freq = freq_df.select_dtypes(include=[np.number]).values.flatten()
                csv_features.extend(numeric_freq)
            except Exception as e:
                pass
        
        if not csv_features:
            return None
        
        csv_features = np.array(csv_features, dtype=np.float32)
        target_dim = getattr(self, 'csv_feature_dim', 64)
        
        # 填充或截断
        if len(csv_features) > target_dim:
            csv_features = csv_features[:target_dim]
        else:
            csv_features = np.pad(csv_features, (0, target_dim - len(csv_features)), 'constant')
        
        return csv_features
    
    def _extract_fault_type(self, folder_name):
        """从文件夹名称提取故障类型"""
        if '_N_' in folder_name or folder_name.startswith('48k_Normal'):
            return 'N'
        elif '_B_' in folder_name or '_B0' in folder_name:
            return 'B'
        elif '_IR_' in folder_name or '_IR0' in folder_name:
            return 'IR'
        elif '_OR_' in folder_name or '_OR0' in folder_name:
            return 'OR'
        else:
            return 'N'
    
    def _augment_sequence(self, sequence):
        """序列数据增强"""
        if not self.use_augmentation:
            return sequence
        
        # 随机噪声
        if np.random.random() < 0.3:
            noise = np.random.normal(0, 0.01, sequence.shape)
            sequence = sequence + noise
        
        # 随机缩放
        if np.random.random() < 0.3:
            scale = np.random.uniform(0.9, 1.1)
            sequence = sequence * scale
        
        # 随机时间偏移
        if np.random.random() < 0.3:
            shift = np.random.randint(-100, 100)
            sequence = np.roll(sequence, shift)
        
        return sequence
    
    def _augment_image(self, image):
        """图像数据增强"""
        if not self.use_augmentation:
            return image
        
        # 随机亮度调整
        if np.random.random() < 0.3:
            brightness = np.random.uniform(0.8, 1.2)
            image = np.clip(image * brightness, 0, 1)
        
        # 随机对比度调整
        if np.random.random() < 0.3:
            contrast = np.random.uniform(0.8, 1.2)
            image = np.clip((image - 0.5) * contrast + 0.5, 0, 1)
        
        return image
    
    def __len__(self):
        return len(self.data_paths)
    
    def __getitem__(self, idx):
        data_path = self.data_paths[idx]
        folder_name = data_path.name
        
        try:
            # 1. 加载图像数据
            time_img_path = data_path / f"{folder_name}_time_domain.png"
            if time_img_path.exists():
                image = cv2.imread(str(time_img_path))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, (224, 224))
                image = image.astype(np.float32) / 255.0
                image = self._augment_image(image)  # 数据增强
                image = np.transpose(image, (2, 0, 1))
            else:
                image = np.zeros((3, 224, 224), dtype=np.float32)
            
            # 2. 加载序列数据
            sequence_path = data_path / f"{folder_name}_raw_data.npy"
            if sequence_path.exists():
                sequence = np.load(sequence_path)
                
                if sequence.ndim > 1:
                    sequence = sequence.flatten()
                
                # 改进的标准化
                sequence = (sequence - np.median(sequence)) / (np.std(sequence) + 1e-8)
                
                # 长度处理
                target_length = 20000
                if len(sequence) > target_length:
                    # 随机裁剪而不是固定裁剪
                    if self.use_augmentation:
                        start_idx = np.random.randint(0, len(sequence) - target_length)
                        sequence = sequence[start_idx:start_idx + target_length]
                    else:
                        sequence = sequence[:target_length]
                else:
                    sequence = np.pad(sequence, (0, target_length - len(sequence)), 'constant')
                
                sequence = self._augment_sequence(sequence)  # 数据增强
            else:
                sequence = np.zeros(20000, dtype=np.float32)
            
            # 3. 加载CSV特征
            csv_features = self._load_csv_features(data_path)
            if csv_features is None:
                csv_features = np.zeros(getattr(self, 'csv_feature_dim', 64), dtype=np.float32)
            else:
                # 标准化
                if self.is_training and hasattr(self, 'scaler'):
                    try:
                        csv_features = self.scaler.transform(csv_features.reshape(1, -1)).flatten()
                    except:
                        csv_features = (csv_features - np.median(csv_features)) / (np.std(csv_features) + 1e-8)
                else:
                    csv_features = (csv_features - np.median(csv_features)) / (np.std(csv_features) + 1e-8)
            
            # 4. 获取标签
            fault_type = self._extract_fault_type(folder_name)
            label = self.fault_mapping[fault_type]
            
            return {
                'image': torch.FloatTensor(image),
                'sequence': torch.FloatTensor(sequence),
                'csv_features': torch.FloatTensor(csv_features),
                'label': torch.LongTensor([label]),
                'folder_name': folder_name
            }
            
        except Exception as e:
            logger.error(f"加载数据失败 {folder_name}: {e}")
            csv_dim = getattr(self, 'csv_feature_dim', 64)
            return {
                'image': torch.zeros(3, 224, 224),
                'sequence': torch.zeros(20000),
                'csv_features': torch.zeros(csv_dim),
                'label': torch.LongTensor([0]),
                'folder_name': folder_name
            }

def create_balanced_dataloader(dataset, batch_size=16, num_workers=0):
    """创建平衡的数据加载器"""
    # 计算类别权重
    labels = []
    for i in range(len(dataset)):
        try:
            sample = dataset[i]
            labels.append(sample['label'].item())
        except:
            labels.append(0)
    
    # 计算采样权重
    class_counts = np.bincount(labels)
    class_weights = 1.0 / class_counts
    sample_weights = [class_weights[label] for label in labels]
    
    # 创建加权采样器
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

def improved_train_fold(model, train_loader, val_loader, device, fold_num, num_epochs=20):
    """改进的训练函数"""
    model.to(device)
    
    # 计算类别权重
    train_labels = []
    for batch in train_loader:
        labels = batch['label']
        if labels.dim() > 1:
            labels = labels.squeeze()
        train_labels.extend(labels.numpy())
    
    class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
    class_weights = torch.FloatTensor(class_weights).to(device)
    
    # 优化器和损失函数
    optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.01, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=0.002, 
        epochs=num_epochs, 
        steps_per_epoch=len(train_loader),
        pct_start=0.3,
        anneal_strategy='cos'
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    
    # 训练历史
    train_losses = []
    val_accuracies = []
    best_val_acc = 0.0
    patience = 0
    max_patience = 5
    
    logger.info(f"开始改进训练 Fold {fold_num}")
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        epoch_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for batch_idx, batch in enumerate(train_loader):
            images = batch['image'].to(device)
            sequences = batch['sequence'].to(device)
            csv_features = batch['csv_features'].to(device)
            labels = batch['label'].to(device)
            
            if labels.dim() > 1:
                labels = labels.squeeze()
            
            optimizer.zero_grad()
            
            outputs = model(images, sequences, csv_features)
            loss = criterion(outputs, labels)
            
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            epoch_loss += loss.item()
            
            # 计算训练准确率
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            
            if batch_idx % 10 == 0:
                current_lr = scheduler.get_last_lr()[0]
                logger.info(f'Fold {fold_num}, Epoch {epoch+1}/{num_epochs}, '
                          f'Batch {batch_idx}, Loss: {loss.item():.4f}, LR: {current_lr:.6f}')
        
        # 验证阶段
        val_acc = evaluate_model(model, val_loader, device)
        
        avg_loss = epoch_loss / len(train_loader)
        train_acc = correct_train / total_train
        train_losses.append(avg_loss)
        val_accuracies.append(val_acc)
        
        logger.info(f'Fold {fold_num}, Epoch {epoch+1}/{num_epochs}, '
                  f'Train Loss: {avg_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}')
        
        # 早停和模型保存
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience = 0
            torch.save(model.state_dict(), f'improved_best_model_fold_{fold_num}.pth')
        else:
            patience += 1
            
        if patience >= max_patience:
            logger.info(f"早停触发，Fold {fold_num} 在 epoch {epoch+1}")
            break
    
    return train_losses, val_accuracies, best_val_acc

def evaluate_model(model, data_loader, device):
    """评估模型"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in data_loader:
            images = batch['image'].to(device)
            sequences = batch['sequence'].to(device)
            csv_features = batch['csv_features'].to(device)
            labels = batch['label'].to(device)
            
            if labels.dim() > 1:
                labels = labels.squeeze()
            
            outputs = model(images, sequences, csv_features)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.numel()
            correct += (predicted == labels).sum().item()
    
    return correct / total

def improved_k_fold_training(data_dir, num_folds=3, num_epochs=20):
    """改进的K折交叉验证训练"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    
    # 确定CSV特征维度
    sample_dataset = ImprovedBearingDataset(data_dir, fold_num=1)
    csv_dim = sample_dataset.csv_feature_dim
    logger.info(f"CSV特征维度: {csv_dim}")
    
    fold_results = []
    
    for fold in range(1, num_folds + 1):
        logger.info(f"\n{'='*50}")
        logger.info(f"开始改进训练 Fold {fold}/{num_folds}")
        logger.info(f"{'='*50}")
        
        # 准备数据
        train_folds = [i for i in range(1, num_folds + 1) if i != fold]
        
        # 创建训练数据集
        train_paths = []
        for fold_idx in train_folds:
            fold_dir = Path(data_dir) / f"fold_{fold_idx:02d}"
            train_paths.extend(list(fold_dir.iterdir()))
        
        val_fold_dir = Path(data_dir) / f"fold_{fold:02d}"
        val_paths = list(val_fold_dir.iterdir())
        
        # 创建数据集
        train_dataset = ImprovedBearingDataset(data_dir, is_training=True, use_augmentation=True)
        train_dataset.data_paths = [p for p in train_paths if p.is_dir()]
        train_dataset.csv_feature_dim = csv_dim
        
        val_dataset = ImprovedBearingDataset(data_dir, is_training=False, use_augmentation=False)
        val_dataset.data_paths = [p for p in val_paths if p.is_dir()]
        val_dataset.scaler = train_dataset.scaler
        val_dataset.csv_feature_dim = csv_dim
        
        # 创建平衡的数据加载器
        train_loader = create_balanced_dataloader(train_dataset, batch_size=8, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0)
        
        # 创建改进模型
        model = ImprovedMHDCNN(csv_input_dim=csv_dim, num_classes=4)
        
        # 训练模型
        train_losses, val_accuracies, best_val_acc = improved_train_fold(
            model, train_loader, val_loader, device, fold, num_epochs
        )
        
        fold_results.append({
            'fold': fold,
            'best_accuracy': best_val_acc,
            'train_losses': train_losses,
            'val_accuracies': val_accuracies
        })
        
        logger.info(f"Fold {fold} 最佳验证准确率: {best_val_acc:.4f}")
    
    # 计算平均结果
    avg_accuracy = np.mean([result['best_accuracy'] for result in fold_results])
    std_accuracy = np.std([result['best_accuracy'] for result in fold_results])
    
    logger.info(f"\n{'='*50}")
    logger.info(f"改进的K折交叉验证结果")
    logger.info(f"{'='*50}")
    logger.info(f"平均准确率: {avg_accuracy:.4f} ± {std_accuracy:.4f}")
    
    return fold_results, avg_accuracy, std_accuracy

def main():
    """主函数"""
    logger.info("开始改进版MHDCNN轴承故障诊断模型训练")
    
    data_dir = "/Users/gsyoung/MBP_documents/code/Matlab/研究生数学建模/建模实现/处理后数据/dataset"
    
    # 改进的K折交叉验证训练
    fold_results, avg_accuracy, std_accuracy = improved_k_fold_training(
        data_dir=data_dir,
        num_folds=3,
        num_epochs=20
    )
    
    # 保存结果
    results = {
        'fold_results': fold_results,
        'average_accuracy': float(avg_accuracy),
        'std_accuracy': float(std_accuracy),
        'model_architecture': 'Improved_MHDCNN',
        'improvements': [
            'SE注意力机制',
            '改进的数据增强',
            '平衡采样',
            '类别权重',
            '标签平滑',
            '梯度裁剪',
            '早停机制',
            'OneCycleLR调度',
            'RobustScaler标准化'
        ]
    }
    
    with open('improved_training_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info("改进训练完成，结果已保存")

if __name__ == "__main__":
    main()
