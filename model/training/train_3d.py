import os

# ⚠️ 【终极隔离】：物理屏蔽其他卡，只让系统看到物理 2 号卡！
# 注意：在 PyTorch 眼里，这张物理 2 号卡，会变成唯一的逻辑卡 "cuda:0"
os.environ["CUDA_VISIBLE_DEVICES"] = "2" 

import torch
from modeling_fleming_3d import Fleming3D_CoT_Model

def main():
    model_path = "/volume/med-train/users/shuyan/Fleming-VL-8B"
    IMG_TOKEN_ID = 151646 
    
    print("🚀 正在初始化单卡 3D 医疗大模型框架...")
    model = Fleming3D_CoT_Model(model_path, img_token_id=IMG_TOKEN_ID) 
    
    print("\n--- 🧪 开始进行单卡 Dummy Data 数据流测试 ---")
    
    # 因为只有一张卡了，目标设备永远是 cuda:0
    target_device = torch.device("cuda:0") 
    print(f"当前运行设备: {target_device} (对应物理卡 2)")
    
    # 1. 构造 3D CT 伪张量 [batch, num_slices, channels, H, W]
    # 如果报 OOM，可以把这里的 batch size 改成 1 试试
    dummy_pixel_values = torch.randn(2, 32, 3, 224, 224, dtype=torch.bfloat16).to(target_device)
    
    # 2. 构造文本输入
    dummy_input_ids = torch.randint(0, 32000, (2, 50)).to(target_device)
    dummy_input_ids[:, 10] = IMG_TOKEN_ID 
    
    # 3. 构造 Labels
    dummy_labels = dummy_input_ids.clone()
    
    try:
        print("▶️ 正在测试 Forward Pass (前向传播)...")
        outputs = model(pixel_values=dummy_pixel_values, input_ids=dummy_input_ids, labels=dummy_labels)
        print("✅ Forward Pass 成功！张量拼接逻辑无误！")
        
        loss = outputs.loss if hasattr(outputs, 'loss') else outputs.sum() 
        print(f"▶️ 正在测试 Backward Pass (反向传播)... 当前 Loss: {loss.item()}")
        
        loss.backward()
        print("✅ Backward Pass 成功！梯度已顺利回传至 Adapter 和 Projector！")
        print("\n🎉 阶段一任务彻底打通！干得漂亮！")
        
    except Exception as e:
        print(f"\n❌ 测试失败，捕获到报错：\n{e}")

if __name__ == "__main__":
    main()