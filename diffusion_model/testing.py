import torch
import matplotlib.pyplot as plt
from model import Unet
from dataloader import load_data
from training import compute_alpha_bar

def generate_from_noise(model, noise, start_step, max_time_steps, device):
    model.eval()

    with torch.no_grad():
        x = noise
        initial_noise = x.clone()

        for t in range(start_step - 1, -1, -1):
            t_tensor = torch.tensor([t], device=device)
            
            # Utilisation de max_time_steps (T) pour respecter le schedule d'entraînement
            alpha_bar_t = compute_alpha_bar(t_tensor, max_time_steps)
            
            if t > 0:
                alpha_bar_t_minus_1 = compute_alpha_bar(torch.tensor([t - 1], device=device), max_time_steps)
            else:
                alpha_bar_t_minus_1 = torch.tensor([1.0], device=device).view(-1, 1, 1, 1)

            alpha_t = alpha_bar_t / alpha_bar_t_minus_1
            alpha_t = torch.clamp(alpha_t, min=1e-5, max=1.0)
            beta_t = 1.0 - alpha_t

            eps_predicted = model(x, t_tensor)

            x = (1.0 / torch.sqrt(alpha_t)) * (x - (beta_t / torch.sqrt(1 - alpha_bar_t)) * eps_predicted)

            if t > 0:
                z = torch.randn_like(x)
                x = x + torch.sqrt(beta_t) * z
                
        # Clamping de rigueur avant la dé-normalisation
        x = torch.clamp(x, -1.0, 1.0)
        x = (x + 1.0) / 2.0 
        initial_noise = (initial_noise + 1.0) / 2.0
        
        img_noise = initial_noise.cpu().squeeze().numpy()
        img_generated = x.cpu().squeeze().numpy()
        
        if start_step == max_time_steps:
            n=2
        else:
            n=3
        
        fig, axes = plt.subplots(1, n, figsize=(8, 4))
        
        axes[0].imshow(img_noise, cmap='gray')
        axes[0].set_title(f"Bruit (T={start_step})")
        axes[0].axis('off')
            
        axes[1].imshow(img_generated, cmap='gray')
        axes[1].set_title("Chiffre Généré (T=0)")
        axes[1].axis('off')
        
        if start_step == max_time_steps:
            plt.show()
        else:
            return fig, axes


def generate_from_default(model, dataloader, start_time, max_time_steps, device):
    model.eval()

    with torch.no_grad():
        for suspect_images, _ in dataloader:
            suspect_images = suspect_images.to(device)
            suspect_images = suspect_images[0].unsqueeze(0) 
            
            suspect_images = suspect_images * 2.0 - 1.0
            
            t_tensor = torch.tensor([start_time], device=device)
            
            eps = torch.randn_like(suspect_images)
            alpha_bar = compute_alpha_bar(t_tensor, max_time_steps)
            
            # Forward process
            noise = torch.sqrt(alpha_bar) * suspect_images + torch.sqrt(1 - alpha_bar) * eps
            
            # Reverse process
            fig, axes = generate_from_noise(model, noise, start_step=start_time, max_time_steps=max_time_steps, device=device)
            
            axes[2].imshow(suspect_images.squeeze().squeeze().cpu().numpy(), cmap='grey')
            axes[2].set_title("Original sample (T=0)")
            axes[2].axis('off')

            plt.show()

            break

if __name__=="__main__":
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    model = Unet(time_emb_dim=128, base_channels=32).to(device)
    model.load_state_dict(torch.load("ddpm_weights_normal.pth", map_location=device))
    
    # Test 1 : Génération pure depuis le bruit total
    #print("Génération depuis le bruit total...")
    #pure_noise = torch.randn((1, 1, 28, 28), device=device)
    #generate_from_noise(model, pure_noise, start_step=500, max_time_steps=500, device=device)

    # Test 2 : Inpainting / Denoising partiel
    print("Dénouement partiel d'une image existante...")
    dataloader = load_data(is_training=False, data_type="normal", is_grayscale=True)
    generate_from_default(model, dataloader, start_time=320, max_time_steps=500, device=device)