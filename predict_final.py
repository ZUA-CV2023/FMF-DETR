import sys
from pathlib import Path
import warnings
import os
import random

warnings.filterwarnings('ignore')

# 1. 【关键】将当前项目路径加入系统路径，确保优先加载本地修改过的 ultralytics
project_root = Path('/root/autodl-tmp/WT-DETR')
sys.path.insert(0, str(project_root))

# 2. 【关键】注册自定义模块 (必须与训练时一致)
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
    
    print("✅ 自定义模块注册成功")
except ImportError as e:
    print(f"❌ 模块注册失败: {e}")
    sys.exit(1)

from ultralytics import RTDETR

def predict_image():
    # 模型路径
    model_path = 'runs/train/exp6/weights/best.pt'
    
    # 3. 指定要预测的图片路径
    # 这里我写了一个逻辑：如果你指定的图片不存在，就自动从验证集里随机挑一张
    target_img = 'dataset/images/val/0000001_02999_d_0000005.jpg'
    
    if not os.path.exists(target_img):
        print(f"⚠️ 指定图片 {target_img} 不存在，正在从验证集中随机选取一张...")
        val_dir = 'dataset/images/val'
        images = [os.path.join(val_dir, f) for f in os.listdir(val_dir) if f.endswith('.jpg')]
        if images:
            target_img = random.choice(images)
        else:
            print("❌ 验证集文件夹是空的！")
            return

    print(f"🚀 正在加载模型: {model_path}")
    print(f"📸 正在预测图片: {target_img}")
    
    # 加载模型
    model = RTDETR(model_path)
    
    # 4. 开始预测
    # save=True 会把结果保存下来
    # conf=0.25 是置信度阈值，只有大于这个分数的框才会被画出来
    results = model.predict(source=target_img, save=True, conf=0.25, line_width=2)
    
    # 5. 告知结果位置
    print("\n" + "="*50)
    print("✅ 预测完成！")
    print(f"结果已保存在: {results[0].save_dir}")
    print("="*50)

if __name__ == '__main__':
    predict_image()