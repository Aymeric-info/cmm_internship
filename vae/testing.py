import torch
import matplotlib.pyplot as plt
from dataloader import load_data
from model import VAE

def inspect_anomalies(model, dataloader, device):
    model.eval()

    with torch.no_grad():
        for suspect_images, _ in dataloader:
            suspect_images = suspect_images.to(device)

            reconstructed_images, mean, logvar = model(suspect_images)

            error_map = torch.abs(suspect_images - reconstructed_images)

            original_img = suspect_images[0].cpu().squeeze()
            reconstructed_img = reconstructed_images[0].cpu().squeeze()
            error_img = error_map[0].cpu().squeeze()
            error_img[error_img<0.5] = 0

            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            axes[0].imshow(original_img, cmap='gray')
            axes[0].set_title("Anomaly (Input)")
            
            axes[1].imshow(reconstructed_img, cmap='gray')
            axes[1].set_title("Reconstructed through VAE")
            
            axes[2].imshow(error_img, cmap='hot')
            axes[2].set_title("Heatmap of the anomaly (mask : pixel > 0.5)")
            
            plt.show()
            
            break 

if __name__=="__main__":
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    model = VAE(embedding_dim=8, input_size=28).to(device)
    model.load_state_dict(torch.load("vae_weights_normal.pth", map_location=device))
    
    anomaly_loader = load_data(is_training=False, data_type="anomaly", is_grayscale=True)
    
    inspect_anomalies(model, anomaly_loader, device)