import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, embedding_dim, input_size):
        super().__init__()
        self.embedding_dim = embedding_dim
        assert (input_size % 4 == 0)

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=8, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )

        self.flatten = nn.Flatten()

        self.linear_mean = nn.Linear(2*input_size**2, self.embedding_dim)   # stride = 2 so output_conv = input_size//4
        self.linear_logvar = nn.Linear(2*input_size**2, self.embedding_dim)

    def forward(self, x):
        out = self.layers(x)
        out = self.flatten(out)
        mean = self.linear_mean(out)
        logvar = self.linear_logvar(out)    # logvar stands for log(std)
        return mean, logvar


class Decoder(nn.Module):
    def __init__(self, embedding_dim, input_size):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.input_size = input_size
        assert (input_size % 4 == 0)

        self.linear = nn.Linear(self.embedding_dim, 2*input_size**2)

        # ConvTranspose2d allows us to upscale the image (7x7 --> 14x14 --> 28x28)
        self.layers = nn.Sequential(
            nn.ConvTranspose2d(in_channels=32, out_channels=8, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=8, out_channels=1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, z):
        out = self.linear(z)
        
        out = out.reshape(-1, 32, self.input_size // 4, self.input_size // 4)
        
        out = self.layers(out)
        return out

class VAE(nn.Module):
    def __init__(self, embedding_dim, input_size):
        super().__init__()
        self.embedding_dim = embedding_dim

        self.encoder = Encoder(embedding_dim, input_size)
        self.decoder = Decoder(embedding_dim, input_size)
    
    def forward(self, x):
        mean, logvar = self.encoder(x)

        std = torch.exp(logvar)
        eps = torch.randn_like(mean)    # sampling of a normal distribution N(0, I)
        z = mean + std*eps

        out = self.decoder(z)
        return out, mean, logvar