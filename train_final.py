#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
"""

import warnings
warnings.filterwarnings('ignore')

import sys
from pathlib import Path

# 添加项目路径
project_root = Path('/root/autodl-tmp/WT-DETR')
sys.path.insert(0, str(project_root))

# 首先注册自定义模块
print("=" * 70)
print("🔧 步骤 1: 注册自定义模块")
print("=" * 70)

try:
    # 导入并注册自定义模块
    from ultralytics.nn.modules.wavelet import WaveletPool, WaveletUnPool
    from ultralytics.nn.modules.freq_enhancement_module import FrequencyEnhancementModule
    from ultralytics.nn.modules.balancing_diffusion_module import BalancingDiffusionModule
    
    import ultralytics.nn.modules as modules
    
    # 注册模块
    modules.WaveletPool = WaveletPool
    modules.WaveletUnPool = WaveletUnPool
    modules.FrequencyEnhancementModule = FrequencyEnhancementModule
    modules.BalancingDiffusionModule = BalancingDiffusionModule
    
    # 更新 __all__
    if hasattr(modules, '__all__'):
        for mod in ['WaveletPool', 'WaveletUnPool', 'FrequencyEnhancementModule', 'BalancingDiffusionModule']:
            if mod not in modules.__all__:
                modules.__all__.append(mod)
    
    print("✅ 自定义模块注册成功!")
    print("   • WaveletPool")
    print("   • WaveletUnPool")
    print("   • FrequencyEnhancementModule")
    print("   • BalancingDiffusionModule")
    
except ImportError as e:
    print(f"❌ 模块导入失败: {e}")
    print("   请确保所有模块文件都已创建")
    print("   需要的文件:")
    print("   - ultralytics/nn/modules/wavelet.py")
    print("   - ultralytics/nn/modules/freq_enhancement_module.py")
    print("   - ultralytics/nn/modules/balancing_diffusion_module.py")
    sys.exit(1)

# 导入 Ultralytics
from ultralytics import RTDETR
import torch


def check_environment():
    """检查训练环境"""
    print("\n" + "=" * 70)
    print("🔍 步骤 2: 检查环境")
    print("=" * 70)
    
    # PyTorch 版本
    print(f"PyTorch 版本: {torch.__version__}")
    
    # CUDA 可用性
    if torch.cuda.is_available():
        print(f"✅ CUDA 可用")
        print(f"   设备数量: {torch.cuda.device_count()}")
        print(f"   当前设备: {torch.cuda.current_device()}")
        print(f"   设备名称: {torch.cuda.get_device_name(0)}")
        print(f"   显存大小: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print(f"⚠️  CUDA 不可用，将使用 CPU 训练（速度较慢）")
    
    # 检查数据集
    dataset_path = project_root / 'dataset' / 'VisDrone.yaml'
    if dataset_path.exists():
        print(f"✅ 数据集配置存在: {dataset_path}")
    else:
        print(f"⚠️  数据集配置不存在: {dataset_path}")
        print(f"   请检查数据集路径")


def get_model_config():
    """获取模型配置"""
    print("\n" + "=" * 70)
    print("📋 步骤 3: 选择模型配置")
    print("=" * 70)
    
    # 检查改进的配置是否存在
    improved_config = project_root / 'ultralytics/cfg/models/rtdetr-improved.yaml'
    original_config = project_root / 'ultralytics/cfg/models/rtdetr-r18-WT-Z.yaml'
    
    if improved_config.exists():
        print(f"✅ 找到改进的配置: {improved_config.name}")
        print("   使用改进的配置 (包含频域增强和平衡扩散)")
        return str(improved_config)
    elif original_config.exists():
        print(f"✅ 找到原始配置: {original_config.name}")
        print("   使用原始配置")
        return str(original_config)
    else:
        print(f"⚠️  未找到配置文件")
        print(f"   将使用默认配置: rtdetr-l.yaml")
        return 'rtdetr-l.yaml'


def train_model(model_config, use_improved=False):
    """训练模型"""
    print("\n" + "=" * 70)
    print("🚀 步骤 4: 开始训练")
    print("=" * 70)
    
    # 创建模型
    print(f"\n创建模型: {model_config}")
    model = RTDETR(model_config)
    
    # 可选：加载预训练权重
    # model.load('path/to/pretrained.pt')
    
    # 训练参数
    train_args = {
        # ===== 数据集 =====
        'data': 'dataset/VisDrone.yaml',
        'cache': False,
        
        # ===== 基础训练参数 =====
        'imgsz': 640,
        'epochs': 300,
        'batch': 16 if torch.cuda.is_available() else 4,  # 根据 GPU 调整
        'workers': 4,
        'device': '0' if torch.cuda.is_available() else 'cpu',
        
        # ===== 优化器 =====
        'optimizer': 'AdamW',
        'lr0': 0.001,
        'lrf': 0.01,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3.0,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        
        # ===== 损失权重 =====
        'box': 7.5,
        'cls': 0.5,
        'dfl': 1.5,
        
        # ===== 数据增强 =====
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
        'degrees': 0.0,
        'translate': 0.1,
        'scale': 0.5,
        'shear': 0.0,
        'perspective': 0.0,
        'flipud': 0.0,
        'fliplr': 0.5,
        'mosaic': 1.0,
        'mixup': 0.0,
        'copy_paste': 0.0,
        
        # ===== 验证和保存 =====
        'val': True,
        'save': True,
        'save_period': 10,
        'patience': 50,
        'plots': True,
        
        # ===== 输出路径 =====
        'project': 'runs/train',
        'name': 'improved_rtdetr' if use_improved else 'exp',
        'exist_ok': False,
        
        # ===== 其他 =====
        'verbose': True,
        'seed': 0,
        'deterministic': True,
        'amp': False,  # 混合精度训练
        'close_mosaic': 10,
        
        # ===== 恢复训练（如果需要）=====
        # 'resume': 'runs/train/exp/weights/last.pt',
    }
    
    # 打印训练配置
    print("\n训练配置:")
    print(f"   数据集: {train_args['data']}")
    print(f"   轮数: {train_args['epochs']}")
    print(f"   批次大小: {train_args['batch']}")
    print(f"   图像尺寸: {train_args['imgsz']}")
    print(f"   设备: {train_args['device']}")
    print(f"   优化器: {train_args['optimizer']}")
    print(f"   初始学习率: {train_args['lr0']}")
    print(f"   混合精度: {train_args['amp']}")
    
    # 开始训练
    print(f"\n🎯 开始训练...\n")
    results = model.train(**train_args)
    
    return results


if __name__ == '__main__':
    try:
        # 1. 检查环境
        check_environment()
        
        # 2. 获取模型配置
        model_config = get_model_config()
        
        # 判断是否使用改进配置
        use_improved = 'improved' in model_config.lower()
        
        # 3. 训练模型
        results = train_model(model_config, use_improved)
        
        # 4. 训练完成
        print("\n" + "=" * 70)
        print("✅ 训练完成!")
        print("=" * 70)
        print("\n结果保存在:")
        print(f"   runs/train/{'improved_rtdetr' if use_improved else 'exp'}/")
        print("\n可以使用以下命令评估模型:")
        print(f"   python -c \"from ultralytics import RTDETR; model = RTDETR('runs/train/{'improved_rtdetr' if use_improved else 'exp'}/weights/best.pt'); model.val()\"")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  训练被用户中断")
    except Exception as e:
        print(f"\n\n❌ 训练出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)