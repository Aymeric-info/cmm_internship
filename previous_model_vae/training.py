import torch
from numpy import around
import torch.optim as optim
import torch.nn.functional as F
from dataloader import load_data
from model import VAE

def elbo(x, reconstructed_x, mean, logvar):
    bce = F.binary_cross_entropy(reconstructed_x, x, reduction="sum")
    kld = -0.5*torch.sum(1 + 2*logvar - mean.pow(2) - torch.exp(2*logvar))
    return kld + bce


def train(dataloader, model, epochs=1000):
    optimizer = optim.Adam(model.parameters())
    model.train()
    epoch_volume = dataloader.__len__()*list(dataloader)[0][0].shape[2]**2

    for epoch in range(epochs):
        epoch_loss = 0.0

        for images, _ in dataloader:
            images = images.to(device)
            reconstructed_images, mean, logvar = model(images)

            loss = elbo(images, reconstructed_images, mean, logvar)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        
        error = around(epoch_loss / epoch_volume, decimals=2)
        
        if epoch%10==0:
            print(f"Epoch : {epoch}, Error : {error}")


if __name__=="__main__":

    if torch.cuda.is_available():
        device = torch.device("cuda") # for NVIDIA
    elif torch.backends.mps.is_available():
        device = torch.device("mps")  # for Mac (m1, m2...)
    else:
        device = torch.device("cpu")  # slow

    
    model = VAE(embedding_dim=8, input_size=28).to(device)
    """    
    # training on normal samples (here 8 in MNIST)
    dataloader = load_data(data_type="normal", is_grayscale=True)
    print(torch.backends.mps.is_available())
    train(dataloader, model)

    save_path = "vae_weights_normal.pth"
    torch.save(model.state_dict(), save_path)
    """
    
    # training on only 10 anomaly samples (here 3 in MNIST)
    model.load_state_dict(torch.load("vae_weights_normal.pth", map_location=device))
    dataloader = load_data(is_training=True, data_type="anomaly", is_grayscale=True)
    train(dataloader, model)

    save_path = "vae_weights_normal_and_anomaly.pth"
    torch.save(model.state_dict(), save_path)