import sys
import os
import cv2
import torch
from pathlib import Path


project_root = Path('/root/autodl-tmp/WT-DETR')
sys.path.insert(0, str(project_root))

from ultralytics.nn.modules.wavelet import WaveletPool, WaveletUnPool
from ultralytics.nn.modules.freq_enhancement_module import FrequencyEnhancementModule
from ultralytics.nn.modules.balancing_diffusion_module import BalancingDiffusionModule
import ultralytics.nn.modules as modules

modules.WaveletPool = WaveletPool
modules.WaveletUnPool = WaveletUnPool
modules.FrequencyEnhancementModule = FrequencyEnhancementModule
modules.BalancingDiffusionModule = BalancingDiffusionModule

from ultralytics import RTDETR


baseline_path = 'runs/train/baseline_exp2/weights/best.pt' 

improved_path = 'runs/train/exp6/weights/best.pt'

test_dir = 'dataset/images/val'

save_dir = 'ablation_results'
os.makedirs(save_dir, exist_ok=True)


print("正在加载模型...")
model_base = RTDETR(baseline_path)
model_ours = RTDETR(improved_path)


test_images = [f for f in os.listdir(test_dir) if f.endswith('.jpg')][:10]

for img_name in test_images:
    img_path = os.path.join(test_dir, img_name)
    

    results_base = model_base(img_path, conf=0.25)[0]
    results_ours = model_ours(img_path, conf=0.25)[0]
    

    plot_base = results_base.plot(labels=False, probs=False)
    plot_ours = results_ours.plot(labels=False, probs=False)
    

    cv2.putText(plot_base, "Baseline (RT-DETR-R18)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(plot_ours, "Ours (FEM + BDM)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    

    combined = cv2.hconcat([plot_base, plot_ours])
    

    cv2.imwrite(os.path.join(save_dir, f"comp_{img_name}"), combined)
    print(f"已生成对比图: comp_{img_name}")

print(f"\n✅ 所有对比图已保存在: {save_dir}")