import onnx
from onnxconverter_common import float16

# 1. 加载刚才生成的标准版模型 (请根据你实际生成的路径修改)
model_path = 'runs/train/exp6/weights/best.onnx'
model = onnx.load(model_path)

print("🚀 正在将 ONNX 模型转换为 FP16 精度（50% 压缩）...")

# 2. 执行半精度转换
model_fp16 = float16.convert_float_to_float16(model)

# 3. 保存压缩后的模型
output_path = 'runs/train/exp6/weights/best_fp16.onnx'
onnx.save(model_fp16, output_path)

import os
old_size = os.path.getsize(model_path) / (1024 * 1024)
new_size = os.path.getsize(output_path) / (1024 * 1024)

print("\n" + "="*50)
print(f"✅ 压缩完成！")
print(f"原始大小 (FP32): {old_size:.2f} MB")
print(f"压缩后大小 (FP16): {new_size:.2f} MB")
print(f"压缩率: {((old_size - new_size) / old_size * 100):.1f}%")
print("="*50)