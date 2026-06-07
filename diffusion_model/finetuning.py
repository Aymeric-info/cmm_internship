import torch
import torch.nn as nn
import math
from model import Unet
from dataloader import load_data
from training import train
from parameters import time_emb_dim, base_channels, time_steps, device, finetuning_epochs, r


class LoRAConv2d(nn.Module):
    def __init__(self, base_conv: nn.Conv2d, r, lora_intensity = 1.0):
        super().__init__()
        self.r = r
        self.lora_intensity = lora_intensity
        
        self.base_conv = base_conv
        self.base_conv.weight.requires_grad = False
        self.base_conv.bias.requires_grad = False

        # building a low rank convolution
        self.lora_A = nn.Conv2d(base_conv.in_channels, r, base_conv.kernel_size, base_conv.stride, base_conv.padding, bias=False)
        self.lora_B = nn.Conv2d(r, base_conv.out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        
        # AI recommendations for initialisation of lora matrix
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        base_out = self.base_conv(x)
        
        if not getattr(self, 'use_lora', True): # do we want to use LoRA adaptation ?
            return base_out
        
        if getattr(self, 'use_cfg', False) and not self.training:  # do we want to use classifier free guidance ?
            # batch splitted in conditioned noise for guidance and uncond noise
            x_uncond, x_cond = x.chunk(2, dim=0) 
            
            lora_cond = self.lora_B(self.lora_A(x_cond))
            
            lora_out = torch.cat([torch.zeros_like(lora_cond), lora_cond], dim=0)   # gathered in the batch
            
            return base_out + (self.lora_intensity / self.r) * lora_out
    
        # for training
        lora_out = self.lora_B(self.lora_A(x))
        return base_out + (self.lora_intensity / self.r) * lora_out


def inject_lora(model, target_modules=["resblock2", "decoder"]):
    for param in model.parameters():
        param.requires_grad = False
        
    for name, module in model.named_modules():
        if any(target in name for target in target_modules):
            if isinstance(module, nn.Conv2d):
                parent_name = name.rsplit('.', 1)[0]
                child_name = name.rsplit('.', 1)[-1]
                parent = model.get_submodule(parent_name)
                
                lora_layer = LoRAConv2d(module, r)
                setattr(parent, child_name, lora_layer)
                
    for name, param in model.named_parameters():
        if "lora_" in name:
            param.requires_grad = True
            
    return model

if __name__=="__main__":    
    model = Unet(time_emb_dim, base_channels, time_steps).to(device)
    model.load_state_dict(torch.load("parameters/ddpm_weights_normal.pth", map_location=device))

    model = inject_lora(model).to(device)

    print(f"Finetuning with LoRA on : {device}")

    dataloader = load_data(is_training=True, data_type="anomaly", is_grayscale=True)
    train(dataloader, model, time_steps, device, finetuning_epochs)

    save_path = "parameters/ddpm_weights_lora.pth"
    torch.save(model.state_dict(), save_path)