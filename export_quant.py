import sys
from pathlib import Path
from ultralytics import RTDETR
import ultralytics.nn.modules as modules
from ultralytics.nn.modules.wavelet import WaveletPool
from ultralytics.nn.modules.freq_enhancement_module import FrequencyEnhancementModule
from ultralytics.nn.modules.balancing_diffusion_module import BalancingDiffusionModule

# 注册模块
modules.WaveletPool = WaveletPool
modules.FrequencyEnhancementModule = FrequencyEnhancementModule
modules.BalancingDiffusionModule = BalancingDiffusionModule

def export_standard():
    model = RTDETR('runs/train/exp6/weights/best.pt')
    print("📦 正在导出标准 FP32 ONNX 模型...")
    
    # 注意：这里关闭 half，提升 opset
    path = model.export(
        format='onnx', 
        imgsz=640, 
        half=False,    # 先不在这里压缩，避开 Bug
        simplify=True, 
        opset=16, 
        device=0
    )
    print(f"✅ 标准版导出成功: {path}")

if __name__ == '__main__':
    export_standard()