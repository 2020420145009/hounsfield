import torch
import torch.nn as nn
from transformers import AutoModel

class PerceiverResampler(nn.Module):
    def __init__(self, dim, depth, dim_head, heads, num_latents):
        super().__init__()
        self.num_latents = num_latents
        self.proj = nn.Linear(dim, dim) 
        self.adapter_3d = PerceiverResampler(
            dim=vision_dim, depth=2, dim_head=64, heads=8, num_latents=self.num_3d_tokens
        )
        
        self.new_projector = nn.Sequential(
            nn.Linear(vision_dim, llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim)
        )
        
        # ⚠️ 【新增代码】：强行把新初始化的权重转为 bfloat16，与底座对齐
        self.adapter_3d.to(torch.bfloat16)
        self.new_projector.to(torch.bfloat16)
        
        self.adapter_3d.requires_grad_(True)
        self.new_projector.requires_grad_(True)
        
        # 下面是你之前的代码
        self.adapter_3d.to("cuda:0")
        self.new_projector.to("cuda:0")
        
    def forward(self, x):
        pooled = self.proj(x.mean(dim=1).mean(dim=1)) 
        return pooled.unsqueeze(1).repeat(1, self.num_latents, 1)

class Fleming3D_CoT_Model(nn.Module):
    def __init__(self, model_path, img_token_id=151646): 
        super().__init__()
        self.img_token_id = img_token_id
        
        print("正在加载 Fleming 底座到单一显卡...")
        # ⚠️ 【关键修改 1】：去掉了 device_map="auto"
        # 直接使用 .to("cuda:0") 把模型扔到这唯一可见的卡上
        self.base_model = AutoModel.from_pretrained(
            model_path, 
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        ).to("cuda:0") 
        
        for param in self.base_model.parameters():
            param.requires_grad = False
            
        vision_dim = self.base_model.config.vision_config.hidden_size
        llm_dim = self.base_model.config.llm_config.hidden_size
        self.num_3d_tokens = 64
        
        self.adapter_3d = PerceiverResampler(
            dim=vision_dim, depth=2, dim_head=64, heads=8, num_latents=self.num_3d_tokens
        )
        
        self.new_projector = nn.Sequential(
            nn.Linear(vision_dim, llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim)
        )
        
        self.adapter_3d.requires_grad_(True)
        self.new_projector.requires_grad_(True)
        
        # ⚠️ 【修复 Dtype 报错】：不仅要对齐设备，还要把精度统一成 bfloat16
        self.adapter_3d.to(device="cuda:0", dtype=torch.bfloat16)
        self.new_projector.to(device="cuda:0", dtype=torch.bfloat16)

    def forward(self, pixel_values, input_ids, labels=None):
        b, num_slices, c, h, w = pixel_values.shape
        
        flat_pixels = pixel_values.view(b * num_slices, c, h, w)
        with torch.no_grad():
            vision_outputs = self.base_model.extract_feature(flat_pixels) 
            
        vision_outputs = vision_outputs.view(b, num_slices, -1, vision_outputs.shape[-1])
        features_3d = self.adapter_3d(vision_outputs) 
        language_tokens = self.new_projector(features_3d) 
        
        inputs_embeds = self.base_model.language_model.get_input_embeddings()(input_ids)
        
        new_inputs_embeds = []
        new_labels = []

        for i in range(b):
            img_idx = (input_ids[i] == self.img_token_id).nonzero(as_tuple=True)[0]

            if len(img_idx) == 0:
                new_inputs_embeds.append(inputs_embeds[i])
                if labels is not None:
                    new_labels.append(labels[i])
                continue

            img_idx = img_idx[0]

            prefix_embeds = inputs_embeds[i, :img_idx]
            suffix_embeds = inputs_embeds[i, img_idx + 1:]
            cur_embeds = torch.cat([prefix_embeds, language_tokens[i], suffix_embeds], dim=0)
            new_inputs_embeds.append(cur_embeds)

            if labels is not None:
                prefix_labels = labels[i, :img_idx]
                suffix_labels = labels[i, img_idx + 1:]
                img_labels = torch.full((self.num_3d_tokens,), -100, dtype=labels.dtype, device=labels.device)
                cur_labels = torch.cat([prefix_labels, img_labels, suffix_labels], dim=0)
                new_labels.append(cur_labels)

        inputs_embeds = torch.stack(new_inputs_embeds, dim=0)
        if labels is not None:
            labels = torch.stack(new_labels, dim=0)

        outputs = self.base_model.language_model(
            inputs_embeds=inputs_embeds,
            labels=labels
        )
        
        return outputs