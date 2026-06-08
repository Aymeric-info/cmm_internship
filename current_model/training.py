import torch
import torch.optim as optim
import torch.nn.functional as F
from dataloader import load_data
from model import Unet
from parameters import time_emb_dim, base_channels, time_steps, device, epochs

# commonly taken alpha_bar
def compute_alpha_bar(t_tensor, time_steps, s=0.008):
    t_ratio = t_tensor.float() / time_steps
    f_t = torch.cos(((t_ratio + s) / (1 + s)) * (torch.pi / 2)).pow(2)
    f_0 = torch.cos(torch.tensor(s / (1 + s) * (torch.pi / 2))).pow(2)
    alpha_bar = f_t / f_0
    return torch.clamp(alpha_bar, min=1e-5, max=1-1e-5).view(-1, 1, 1, 1)


def train(dataloader, model, time_steps, device, epochs):
    optimizer = optim.Adam(model.parameters(), lr=1e-3)    
    model.train()
    num_batches = len(dataloader)

    for epoch in range(epochs):
        epoch_loss = 0.0

        for images, _ in dataloader:
            batch_size = images.shape[0]
            images = images.to(device)
            images = images*2 - 1   # centering on 0
            
            t = torch.randint(low=0, high=time_steps, size=(batch_size,), device=device)
            
            eps = torch.randn_like(images)

            alpha_bar = compute_alpha_bar(t, time_steps)
            x = torch.sqrt(alpha_bar) * images + torch.sqrt(1 - alpha_bar) * eps
            
            eps_predicted = model(x, t)
            loss = F.mse_loss(eps_predicted, eps)   # noise prediction
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        
        average_loss = epoch_loss / num_batches
        
        print(f"Epoch : {epoch} | Average Loss : {average_loss:.3f}")


if __name__=="__main__":
    
    print(f"Training on : {device}")

    model = Unet(time_emb_dim=time_emb_dim, base_channels=base_channels, time_steps=time_steps).to(device)
    
    dataloader = load_data(is_training=True, data_type="normal", is_grayscale=True)
    train(dataloader, model, time_steps, device, epochs)

    save_path = "parameters/ddpm_weights_normal.pth"
    torch.save(model.state_dict(), save_path)