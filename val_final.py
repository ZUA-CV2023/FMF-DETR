import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 1. 【关键】将当前项目路径加入系统路径，确保优先加载本地修改过的 ultralytics
project_root = Path('/root/autodl-tmp/WT-DETR')
sys.path.insert(0, str(project_root))

# 2. 【关键】注册自定义模块 (必须与训练时一致，否则无法加载权重)
try:
    from ultralytics.nn.modules.wavelet import WaveletPool, WaveletUnPool
    from ultralytics.nn.modules.freq_enhancement_module import FrequencyEnhancementModule
    from ultralytics.nn.modules.balancing_diffusion_module import BalancingDiffusionModule
    import ultralytics.nn.modules as modules
    
    # 重新注册
    modules.WaveletPool = WaveletPool
    modules.WaveletUnPool = WaveletUnPool
    modules.FrequencyEnhancementModule = FrequencyEnhancementModule
    modules.BalancingDiffusionModule = BalancingDiffusionModule
    
    print("✅ 自定义模块注册成功，准备加载模型...")
except ImportError as e:
    print(f"❌ 模块注册失败: {e}")
    sys.exit(1)

from ultralytics import RTDETR

def validate_model():
    # 3. 加载你训练好的最佳权重
    model_path = 'runs/train/exp6/weights/best.pt'
    
    print(f"正在加载模型: {model_path}")
    model = RTDETR(model_path)
    
    # 4. 开始验证
    # batch=1 用于测试真实的推理速度 (FPS)
    # split='val' 使用验证集，或者 'test' 使用测试集
    print("开始评估...")
    metrics = model.val(
        data='dataset/VisDrone.yaml',
        batch=1,       # 单张推理，为了测 FPS
        imgsz=640,     # 图像尺寸
        conf=0.001,    # 置信度阈值
        iou=0.6,       # NMS IOU 阈值
        device='0',    # 使用 GPU
        split='val'    # 使用验证集
    )
    
    # 5. 打印结果
    print("\n" + "="*50)
    print(f"mAP50: {metrics.box.map50:.5f}")
    print(f"mAP50-95: {metrics.box.map:.5f}")
    
    # 获取推理速度
    speed = metrics.speed
    total_time_ms = speed['preprocess'] + speed['inference'] + speed['postprocess']
    fps = 1000.0 / total_time_ms
    
    print(f"推理耗时 (每张): {total_time_ms:.2f} ms")
    print(f"FPS (帧率): {fps:.2f}")
    print("="*50)

if __name__ == '__main__':
    validate_model()