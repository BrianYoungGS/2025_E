#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MHDCNN项目依赖安装脚本
自动检测并安装所需的Python包
"""

import subprocess
import sys
import importlib
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 必需的依赖包列表
REQUIRED_PACKAGES = {
    'torch': 'torch',
    'torchvision': 'torchvision', 
    'numpy': 'numpy',
    'pandas': 'pandas',
    'scikit-learn': 'scikit-learn',
    'matplotlib': 'matplotlib',
    'seaborn': 'seaborn',
    'cv2': 'opencv-python',
    'PIL': 'Pillow',
    'pathlib': None,  # 内置模块
    're': None,       # 内置模块
    'json': None,     # 内置模块
    'logging': None,  # 内置模块
    'os': None,       # 内置模块
    'warnings': None  # 内置模块
}

def check_package(package_name):
    """检查包是否已安装"""
    try:
        importlib.import_module(package_name)
        return True
    except ImportError:
        return False

def install_package(package_name):
    """安装Python包"""
    try:
        logger.info(f"正在安装 {package_name}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        logger.info(f"✅ {package_name} 安装成功")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ {package_name} 安装失败: {e}")
        return False

def install_pytorch_cpu():
    """安装CPU版本的PyTorch"""
    logger.info("检测到系统可能不支持CUDA，安装CPU版本的PyTorch...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "torch", "torchvision", "torchaudio", 
            "--index-url", "https://download.pytorch.org/whl/cpu"
        ])
        logger.info("✅ PyTorch CPU版本安装成功")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ PyTorch安装失败: {e}")
        return False

def main():
    """主安装函数"""
    logger.info("🚀 开始检查和安装MHDCNN项目依赖...")
    
    missing_packages = []
    
    # 检查所有依赖
    for import_name, package_name in REQUIRED_PACKAGES.items():
        if package_name is None:  # 内置模块
            logger.info(f"✅ {import_name} (内置模块)")
            continue
            
        if check_package(import_name):
            logger.info(f"✅ {import_name} 已安装")
        else:
            logger.warning(f"❌ {import_name} 未安装")
            missing_packages.append((import_name, package_name))
    
    if not missing_packages:
        logger.info("🎉 所有依赖都已安装！")
        return True
    
    # 安装缺失的包
    logger.info(f"📦 需要安装 {len(missing_packages)} 个包...")
    
    failed_packages = []
    
    for import_name, package_name in missing_packages:
        if import_name == 'torch':
            # 特殊处理PyTorch安装
            if not install_pytorch_cpu():
                failed_packages.append(package_name)
        else:
            if not install_package(package_name):
                failed_packages.append(package_name)
    
    # 验证安装
    logger.info("🔍 验证安装结果...")
    all_success = True
    
    for import_name, package_name in missing_packages:
        if check_package(import_name):
            logger.info(f"✅ {import_name} 验证成功")
        else:
            logger.error(f"❌ {import_name} 验证失败")
            all_success = False
    
    if all_success:
        logger.info("🎉 所有依赖安装成功！")
        logger.info("\n" + "="*50)
        logger.info("✨ 现在可以运行MHDCNN训练了！")
        logger.info("执行以下命令开始训练:")
        logger.info("python train_mhdcnn.py")
        logger.info("="*50)
        return True
    else:
        logger.error("❌ 部分依赖安装失败")
        if failed_packages:
            logger.error(f"失败的包: {', '.join(failed_packages)}")
            logger.info("\n手动安装命令:")
            for pkg in failed_packages:
                logger.info(f"pip install {pkg}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
