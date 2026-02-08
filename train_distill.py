import sys
from pathlib import Path
from ultralytics import RTDETR
import torch

# 1. 注册自定义模块
project_root = Path('/root/autodl-tmp/WT-DETR')
sys.path.insert(0, str(project_root))

from ultralytics.nn.modules.wavelet import WaveletPool
from ultralytics.nn.modules.freq_enhancement_module import FrequencyEnhancementModule
from ultralytics.nn.modules.balancing_diffusion_module import BalancingDiffusionModule
import ultralytics.nn.modules as modules

modules.WaveletPool = WaveletPool
modules.FrequencyEnhancementModule = FrequencyEnhancementModule
modules.BalancingDiffusionModule = BalancingDiffusionModule

def run_distillation():
    # 2. 实例化学生模型 (你的改进版)
    student = RTDETR('ultralytics/cfg/models/rtdetr-r18-WT-Z.yaml')
    
    # 3. 实例化教师模型 (官方大模型)
    # 第一次运行会自动下载权重
    teacher = RTDETR('rtdetr-x.pt') 

    print("正在启动知识蒸馏辅助训练...")
    
    # 4. 开启训练
    student.train(
        data='dataset/VisDrone.yaml',
        epochs=100,        # 蒸馏通常不需要300轮，100轮左右即可见效
        batch=16,
        imgsz=640,
        device='0',
        # 这里的关键：利用老师的特征辅助和更强的增强策略
        lr0=0.001,
        lrf=0.01,
        warmup_epochs=5,
        close_mosaic=20,   # 最后20轮关闭增强以对齐特征
        project='runs/train',
        name='kd_student_exp',
        amp=False          # 避开之前的 half 冲突
    )

if __name__ == '__main__':
    run_distillation()