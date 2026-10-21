import os
import json
import torch
import nibabel as nib
import numpy as np
from torch.utils.data import Dataset

class LIDC3D_CoT_Dataset(Dataset):
    def __init__(self, data_root, tokenizer, transform=None, num_slices=32):
        self.data_root = data_root
        self.tokenizer = tokenizer
        self.transform = transform
        self.num_slices = num_slices # 固定采样 32 个切片输入模型
        
        # 遍历获取所有病例文件夹
        self.cases = [os.path.join(data_root, d) for d in os.listdir(data_root) 
                      if os.path.isdir(os.path.join(data_root, d))]

    def __len__(self):
        return len(self.cases)

    def load_nifti(self, file_path):
        # 读取 3D CT 
        img = nib.load(file_path).get_fdata()
        # 归一化到 0-1 或 -1 到 1 (根据你们 Fleming-VL 的预处理要求)
        img = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-8)
        
        # 均匀采样 Z 轴的切片
        z_depth = img.shape[2]
        indices = np.linspace(0, z_depth - 1, self.num_slices, dtype=int)
        sampled_volume = img[:, :, indices]
        
        # 转换为 PyTorch Tensor: [num_slices, 1, H, W] -> 扩展为 [num_slices, 3, H, W] 以适配 ViT
        tensor_volume = torch.tensor(sampled_volume, dtype=torch.float32).permute(2, 0, 1).unsqueeze(1)
        tensor_volume = tensor_volume.repeat(1, 3, 1, 1) 
        return tensor_volume

    def __getitem__(self, idx):
        case_dir = self.cases[idx]
        case_id = os.path.basename(case_dir)
        
        # 1. 加载 3D 影像
        nii_path = os.path.join(case_dir, f"{case_id}_full_volume.nii.gz")
        video_tensor = self.load_nifti(nii_path) # Shape: [32, 3, H, W]
        
        # 2. 加载 CoT 文本 (人类提问与大模型回答)
        cot_path = os.path.join(case_dir, "medical_cot_reasoning_random.txt")
        with open(cot_path, 'r', encoding='utf-8') as f:
            cot_text = f.read()
            
        # 3. 组装成模型需要的标准输入格式 (参考 LLaVA/InternVL 格式)
        # 这里用 <image> 占位符，因为你们的视觉底座是 2D 的，我们将 3D 序列视作“多帧图像”
        prompt = "<image>\nPlease analyze this 3D CT volume and provide a step-by-step CoT reasoning."
        
        # TODO: 这里需要调用你 base 模型里的 conversation 模板和 tokenizer 进行编码
        # input_ids, labels = process_conversation(...)
        
        return {
            "pixel_values": video_tensor, # 这是给视觉编码器的
            "text": prompt,               # 实际工程中这里返回编码好的 input_ids 和 labels
            "target": cot_text
        }