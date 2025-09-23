#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
轴承故障诊断模型配置文件
定义各种实验配置和超参数
"""

from pathlib import Path

# 基础路径配置
BASE_DIR = Path("/Users/gsyoung/MBP_documents/code/Matlab/研究生数学建模/建模实现")
DATA_DIR = BASE_DIR / "处理后数据"
MODEL_DIR = BASE_DIR / "model"
OUTPUT_DIR = BASE_DIR / "outputs"

# 创建输出目录
OUTPUT_DIR.mkdir(exist_ok=True)

# 数据集配置
DATASET_CONFIGS = {
    # 源域配置（用于预训练）
    'source_domain': {
        'name': '12kHz驱动端源域数据',
        'filter': {
            'fault_types': ['N', 'B', 'IR', 'OR'],
            'sampling_freqs': ['12kHz'],
            'sensor_types': ['DE'],
            'data_type': 'segment',
            'loads': [0, 1, 2, 3]
        },
        'preprocessing': {
            'normalize': True,
            'normalize_method': 'standard',
            'augment': True,
            'augment_methods': ['noise', 'scaling'],
            'noise_level': 0.01,
            'balance': True,
            'balance_method': 'oversample'
        }
    },
    
    # 目标域1：不同传感器位置
    'target_domain_sensor': {
        'name': '12kHz风扇端目标域数据',
        'filter': {
            'fault_types': ['N', 'B', 'IR', 'OR'],
            'sampling_freqs': ['12kHz'],
            'sensor_types': ['FE'],
            'data_type': 'segment',
            'loads': [0, 1, 2, 3]
        },
        'preprocessing': {
            'normalize': True,
            'normalize_method': 'standard',
            'augment': False,  # 目标域通常不做数据增强
            'balance': False
        }
    },
    
    # 目标域2：不同采样频率
    'target_domain_frequency': {
        'name': '48kHz目标域数据',
        'filter': {
            'fault_types': ['N', 'B', 'IR', 'OR'],
            'sampling_freqs': ['48kHz'],
            'data_type': 'raw',
            'loads': [0, 1, 2, 3]
        },
        'preprocessing': {
            'normalize': True,
            'normalize_method': 'standard',
            'augment': False,
            'balance': False
        }
    },
    
    # 混合域配置
    'mixed_domain': {
        'name': '混合域数据',
        'filter': {
            'fault_types': ['N', 'B', 'IR', 'OR'],
            'data_type': 'both',
            'loads': [0, 1, 2, 3]
        },
        'preprocessing': {
            'normalize': True,
            'normalize_method': 'robust',  # 混合域使用鲁棒标准化
            'augment': True,
            'augment_methods': ['noise'],
            'noise_level': 0.005,
            'balance': True,
            'balance_method': 'oversample'
        }
    }
}

# 特征工程配置
FEATURE_CONFIGS = {
    'basic': {
        'time_domain': True,
        'frequency_domain': True,
        'wavelet': False,
        'feature_selection': True,
        'selection_method': 'univariate',
        'n_features': 50
    },
    
    'advanced': {
        'time_domain': True,
        'frequency_domain': True,
        'wavelet': True,
        'feature_selection': True,
        'selection_method': 'pca',
        'n_features': 100
    },
    
    'raw_signal': {
        'use_raw_signal': True,
        'signal_length': 2048,
        'normalization': 'standard'
    }
}

# 模型配置
MODEL_CONFIGS = {
    # 传统机器学习模型
    'svm': {
        'model_type': 'svm',
        'params': {
            'C': 1.0,
            'kernel': 'rbf',
            'gamma': 'scale'
        },
        'requires_feature_extraction': True
    },
    
    'random_forest': {
        'model_type': 'random_forest',
        'params': {
            'n_estimators': 100,
            'max_depth': None,
            'random_state': 42
        },
        'requires_feature_extraction': True
    },
    
    'xgboost': {
        'model_type': 'xgboost',
        'params': {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'random_state': 42
        },
        'requires_feature_extraction': True
    },
    
    # 深度学习模型
    'cnn_1d': {
        'model_type': 'cnn_1d',
        'params': {
            'input_length': 2048,
            'n_filters': [32, 64, 128],
            'kernel_sizes': [3, 3, 3],
            'pool_sizes': [2, 2, 2],
            'dense_units': [128, 64],
            'dropout_rate': 0.3,
            'learning_rate': 0.001
        },
        'requires_feature_extraction': False
    },
    
    'lstm': {
        'model_type': 'lstm',
        'params': {
            'input_length': 2048,
            'lstm_units': [64, 32],
            'dense_units': [64],
            'dropout_rate': 0.3,
            'learning_rate': 0.001
        },
        'requires_feature_extraction': False
    },
    
    'transformer': {
        'model_type': 'transformer',
        'params': {
            'input_length': 2048,
            'd_model': 128,
            'n_heads': 8,
            'n_layers': 4,
            'ff_dim': 256,
            'dropout_rate': 0.1,
            'learning_rate': 0.0001
        },
        'requires_feature_extraction': False
    }
}

# 迁移学习配置
TRANSFER_LEARNING_CONFIGS = {
    'fine_tuning': {
        'method': 'fine_tuning',
        'freeze_layers': 0.5,  # 冻结前50%的层
        'learning_rate': 0.0001,  # 较小的学习率
        'epochs': 50
    },
    
    'feature_extraction': {
        'method': 'feature_extraction',
        'freeze_layers': 0.8,  # 冻结前80%的层
        'learning_rate': 0.001,
        'epochs': 30
    },
    
    'domain_adaptation': {
        'method': 'domain_adaptation',
        'lambda_domain': 0.1,  # 域对抗损失权重
        'learning_rate': 0.001,
        'epochs': 100
    }
}

# 训练配置
TRAINING_CONFIGS = {
    'basic': {
        'batch_size': 32,
        'epochs': 100,
        'validation_split': 0.2,
        'early_stopping': {
            'patience': 10,
            'monitor': 'val_accuracy'
        },
        'learning_rate_schedule': {
            'type': 'step',
            'step_size': 30,
            'gamma': 0.1
        }
    },
    
    'advanced': {
        'batch_size': 64,
        'epochs': 200,
        'validation_split': 0.2,
        'early_stopping': {
            'patience': 15,
            'monitor': 'val_accuracy'
        },
        'learning_rate_schedule': {
            'type': 'cosine_annealing',
            'T_max': 200
        },
        'data_augmentation': True
    },
    
    'quick_test': {
        'batch_size': 16,
        'epochs': 10,
        'validation_split': 0.2,
        'early_stopping': {
            'patience': 5,
            'monitor': 'val_loss'
        }
    }
}

# 评估配置
EVALUATION_CONFIGS = {
    'metrics': [
        'accuracy',
        'precision',
        'recall',
        'f1_score',
        'confusion_matrix',
        'classification_report'
    ],
    
    'cross_validation': {
        'cv_folds': 5,
        'stratified': True,
        'random_state': 42
    },
    
    'visualization': {
        'plot_confusion_matrix': True,
        'plot_training_history': True,
        'plot_feature_importance': True,
        'plot_tsne': True
    }
}

# 实验配置（预定义的实验组合）
EXPERIMENT_CONFIGS = {
    'experiment_1': {
        'name': '基础分类实验',
        'description': '使用12kHz DE数据进行基础四分类',
        'dataset': 'source_domain',
        'features': 'basic',
        'model': 'random_forest',
        'training': 'basic'
    },
    
    'experiment_2': {
        'name': '深度学习分类实验',
        'description': '使用CNN进行原始信号分类',
        'dataset': 'source_domain',
        'features': 'raw_signal',
        'model': 'cnn_1d',
        'training': 'advanced'
    },
    
    'experiment_3': {
        'name': '传感器迁移实验',
        'description': '从DE传感器迁移到FE传感器',
        'source_dataset': 'source_domain',
        'target_dataset': 'target_domain_sensor',
        'features': 'advanced',
        'model': 'cnn_1d',
        'transfer_learning': 'fine_tuning',
        'training': 'advanced'
    },
    
    'experiment_4': {
        'name': '频率迁移实验',
        'description': '从12kHz迁移到48kHz',
        'source_dataset': 'source_domain',
        'target_dataset': 'target_domain_frequency',
        'features': 'raw_signal',
        'model': 'transformer',
        'transfer_learning': 'domain_adaptation',
        'training': 'advanced'
    },
    
    'experiment_5': {
        'name': '混合域实验',
        'description': '使用多域数据进行联合训练',
        'dataset': 'mixed_domain',
        'features': 'advanced',
        'model': 'xgboost',
        'training': 'basic'
    }
}

# 故障类型映射
FAULT_LABELS = {
    0: 'Normal',      # 正常
    1: 'Ball',        # 滚动体故障
    2: 'Inner Race',  # 内圈故障
    3: 'Outer Race'   # 外圈故障
}

FAULT_COLORS = {
    'Normal': 'green',
    'Ball': 'orange',
    'Inner Race': 'red',
    'Outer Race': 'blue'
}

# 默认配置
DEFAULT_CONFIG = {
    'dataset': 'source_domain',
    'features': 'basic',
    'model': 'random_forest',
    'training': 'basic',
    'evaluation': EVALUATION_CONFIGS,
    'random_state': 42,
    'verbose': True
}

def get_experiment_config(experiment_name):
    """
    获取指定实验的完整配置
    
    Args:
        experiment_name: 实验名称
        
    Returns:
        完整的实验配置字典
    """
    if experiment_name not in EXPERIMENT_CONFIGS:
        raise ValueError(f"未知的实验配置: {experiment_name}")
    
    exp_config = EXPERIMENT_CONFIGS[experiment_name].copy()
    
    # 填充具体配置
    config = {
        'experiment': exp_config,
        'dataset': DATASET_CONFIGS.get(exp_config.get('dataset')),
        'features': FEATURE_CONFIGS.get(exp_config.get('features')),
        'model': MODEL_CONFIGS.get(exp_config.get('model')),
        'training': TRAINING_CONFIGS.get(exp_config.get('training')),
        'evaluation': EVALUATION_CONFIGS
    }
    
    # 如果是迁移学习实验，添加相关配置
    if 'transfer_learning' in exp_config:
        config['transfer_learning'] = TRANSFER_LEARNING_CONFIGS.get(exp_config['transfer_learning'])
        config['source_dataset'] = DATASET_CONFIGS.get(exp_config.get('source_dataset'))
        config['target_dataset'] = DATASET_CONFIGS.get(exp_config.get('target_dataset'))
    
    return config

def list_available_configs():
    """
    列出所有可用的配置
    """
    print("可用的配置类型:")
    print(f"  数据集配置: {list(DATASET_CONFIGS.keys())}")
    print(f"  特征配置: {list(FEATURE_CONFIGS.keys())}")
    print(f"  模型配置: {list(MODEL_CONFIGS.keys())}")
    print(f"  训练配置: {list(TRAINING_CONFIGS.keys())}")
    print(f"  迁移学习配置: {list(TRANSFER_LEARNING_CONFIGS.keys())}")
    print(f"  实验配置: {list(EXPERIMENT_CONFIGS.keys())}")

if __name__ == "__main__":
    # 演示配置使用
    print("轴承故障诊断配置文件")
    print("=" * 50)
    
    list_available_configs()
    
    print("\n实验配置详情:")
    for exp_name, exp_config in EXPERIMENT_CONFIGS.items():
        print(f"\n{exp_name}:")
        print(f"  名称: {exp_config['name']}")
        print(f"  描述: {exp_config['description']}")
        
        # 获取完整配置示例
        try:
            full_config = get_experiment_config(exp_name)
            print(f"  数据集: {full_config['dataset']['name'] if full_config['dataset'] else 'N/A'}")
            print(f"  模型类型: {full_config['model']['model_type'] if full_config['model'] else 'N/A'}")
        except Exception as e:
            print(f"  配置错误: {e}")
