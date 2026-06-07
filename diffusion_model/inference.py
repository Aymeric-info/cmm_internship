import torch
import matplotlib.pyplot as plt
from model import Unet
from dataloader import load_data
from training import compute_alpha_bar
from finetuning import LoRAConv2d, inject_lora
from parameters import time_emb_dim, base_channels, device, time_steps, start_time

def set_lora_mode(model, use_lora, use_cfg):
    for module in model.modules():
        if isinstance(module, LoRAConv2d):
            module.use_lora = use_lora
            module.use_cfg = use_cfg

def compute_ddpm_step(x, eps_predicted, t, max_time_steps, device):
    t_tensor = torch.tensor([t], device=device)
    
    alpha_bar = compute_alpha_bar(t_tensor, max_time_steps)
    if t > 0:
        alpha_bar_previous = compute_alpha_bar(torch.tensor([t - 1], device=device), max_time_steps)
    else:
        alpha_bar_previous = torch.tensor([1.0], device=device).view(-1, 1, 1, 1)

    alpha = alpha_bar / alpha_bar_previous
    alpha = torch.clamp(alpha, min=1e-5, max=1.0)   # security to avoid numerical issues
    beta = 1.0 - alpha

    x_prev = (1.0 / torch.sqrt(alpha)) * (x - (beta / torch.sqrt(1 - alpha_bar)) * eps_predicted)

    if t > 0:
        z = torch.randn_like(x)
        x_prev = x_prev + torch.sqrt(beta) * z
        
    return x_prev

def generate(model, initial_x, start_step, max_time_steps, device, use_lora=True, cfg_scale=3.0):
    """
    - use_lora = False -> Model without LoRA
    - use_lora = True, cfg_scale = 1.0 -> LoRA and no guidance
    - use_lora = True, cfg_scale > 1.0 -> LoRA and Classifier-Free Guidance
    """
    model.eval()
    
    use_cfg = use_lora and cfg_scale > 1.0
    set_lora_mode(model, use_lora=use_lora, use_cfg=use_cfg)

    with torch.no_grad():
        x = initial_x.clone()

        for t in range(start_step - 1, -1, -1):
            if use_cfg:
                # duplication to then split uncond noise from conditionned noise
                x_in = torch.cat([x, x], dim=0)
                t_in = torch.full((x_in.shape[0],), t, device=device, dtype=torch.long)
                
                eps_preds = model(x_in, t_in)
                eps_uncond, eps_cond = eps_preds.chunk(2, dim=0)
                
                eps_predicted = eps_uncond + cfg_scale * (eps_cond - eps_uncond)
            else:
                t_tensor = torch.tensor([t], device=device)
                eps_predicted = model(x, t_tensor)

            x = compute_ddpm_step(x, eps_predicted, t, max_time_steps, device)
            
    x = torch.clamp(x, -1.0, 1.0)
    x = (x + 1.0) / 2.0 
    
    return x

def run_comparative_inference(model, dataloader, start_time, max_time_steps, device):
    """compares LoRA/CFG."""
    model.eval()

    with torch.no_grad():
        for suspect_images, _ in dataloader:
            suspect_images = suspect_images.to(device)
            suspect_images = suspect_images[0].unsqueeze(0) 
            suspect_images_norm = suspect_images * 2.0 - 1.0 
            
            # Noising until start_time
            t_tensor = torch.tensor([start_time], device=device)
            eps = torch.randn_like(suspect_images_norm)
            alpha_bar = compute_alpha_bar(t_tensor, max_time_steps)
            noise_x = torch.sqrt(alpha_bar) * suspect_images_norm + torch.sqrt(1 - alpha_bar) * eps
            
            # generate imgs
            img_base = generate(model, noise_x, start_time, max_time_steps, device, use_lora=False)
            img_lora = generate(model, noise_x, start_time, max_time_steps, device, use_lora=True, cfg_scale=1.0)
            img_cfg = generate(model, noise_x, start_time, max_time_steps, device, use_lora=True, cfg_scale=4.0)

            # PLOT
            fig, axes = plt.subplots(1, 5, figsize=(15, 3))
            
            axes[0].imshow(suspect_images.cpu().squeeze().numpy(), cmap='gray')
            axes[0].set_title("Original (T=0)")
            axes[0].axis('off')

            axes[1].imshow(((noise_x.cpu() + 1)/2).squeeze().numpy(), cmap='gray')
            axes[1].set_title(f"Noised (T={start_time})")
            axes[1].axis('off')
            
            axes[2].imshow(img_base.cpu().squeeze().numpy(), cmap='gray')
            axes[2].set_title("Base (no LoRA)")
            axes[2].axis('off')
            
            axes[3].imshow(img_lora.cpu().squeeze().numpy(), cmap='gray')
            axes[3].set_title("LoRA (w=1)")
            axes[3].axis('off')
            
            axes[4].imshow(img_cfg.cpu().squeeze().numpy(), cmap='gray')
            axes[4].set_title("LoRA+CFG (w=4)")
            axes[4].axis('off')

            plt.tight_layout()
            plt.show()
            break

if __name__=="__main__":    
    model = Unet(time_emb_dim, base_channels, time_steps)
    model = inject_lora(model)
    model.load_state_dict(torch.load("parameters/ddpm_weights_lora.pth", map_location=device))
    model.to(device)
    
    dataloader = load_data(is_training=False, data_type="normal", is_grayscale=True)

    run_comparative_inference(model, dataloader, start_time, time_steps, device)