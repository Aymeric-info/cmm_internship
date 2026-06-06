import torch
import torch.nn as nn
import math
"""
class SelfAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        # Utilisation de 4 têtes d'attention pour équilibrer coût et performance
        self.mha = nn.MultiheadAttention(embed_dim=channels, num_heads=4, batch_first=True)
        self.ln = nn.LayerNorm([channels])

    def forward(self, x):
        b, c, h, w = x.shape
        # Passage au format séquence (B, L, C) avec L = H * W
        x_unflatten = x.permute(0, 2, 3, 1).view(b, h * w, c)
        x_norm = self.ln(x_unflatten)
        
        attn_out, _ = self.mha(x_norm, x_norm, x_norm)
        attn_out = attn_out + x_unflatten  # Connexion résiduelle
        
        # Restructuration au format tensoriel standard (B, C, H, W)
        return attn_out.view(b, h, w, c).permute(0, 3, 1, 2)

"""
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.norm1 = nn.GroupNorm(num_groups=8, num_channels=in_channels)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.norm2 = nn.GroupNorm(num_groups=8, num_channels=out_channels)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        # Projection temporelle pour AdaGN : génère scale + shift (2 * out_channels)
        self.time_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, 2 * out_channels)
        )
        
        if in_channels != out_channels:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual_conv = nn.Identity()

    def forward(self, x, t_emb):
        out = self.norm1(x)
        out = self.act1(out)
        out = self.conv1(out)

        # Extraction des paramètres de modulation AdaGN
        t_scale_shift = self.time_proj(t_emb).unsqueeze(-1).unsqueeze(-1)
        scale, shift = torch.chunk(t_scale_shift, 2, dim=1)
        
        # Seconde étape intégrant la normalisation conditionnelle
        out = self.norm2(out)
        out = out * (1 + scale) + shift
        out = self.act2(out)
        out = self.conv2(out)

        return out + self.residual_conv(x)


class Encoder(nn.Module):
    def __init__(self, time_emb_dim, base_channels=32):
        super().__init__()
        self.input_conv = nn.Conv2d(in_channels=1, out_channels=base_channels, kernel_size=3, padding=1)
        self.resblock1 = ResBlock(in_channels=base_channels, out_channels=base_channels, time_emb_dim=time_emb_dim)
        self.resblock2 = ResBlock(in_channels=base_channels, out_channels=base_channels * 2, time_emb_dim=time_emb_dim)

        self.pool1 = nn.MaxPool2d(kernel_size=2, padding=0)
        self.pool2 = nn.MaxPool2d(kernel_size=2, padding=0)

    def forward(self, x, t_emb):
        skip_connections = []
        
        out = self.input_conv(x)
        skip_connections.append(out)     # skip[-3] : 28x28, base_channels canaux

        out = self.resblock1(out, t_emb)
        skip_connections.append(out)     # skip[-2] : 28x28, base_channels canaux
        out = self.pool1(out)            # 14x14

        out = self.resblock2(out, t_emb)
        skip_connections.append(out)     # skip[-1] : 14x14, base_channels * 2 canaux
        out = self.pool2(out)            # 7x7, base_channels * 2 canaux

        return out, skip_connections


class Decoder(nn.Module):
    def __init__(self, time_emb_dim, base_channels=32):
        super().__init__()
  
        self.upconv1 = nn.ConvTranspose2d(base_channels * 2, base_channels * 2, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.resblock1 = ResBlock(in_channels=base_channels * 4, out_channels=base_channels * 2, time_emb_dim=time_emb_dim)
        
        self.upconv2 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.resblock2 = ResBlock(in_channels=base_channels * 2, out_channels=base_channels, time_emb_dim=time_emb_dim)
        
        self.output_conv = nn.Sequential(
            nn.Conv2d(in_channels=base_channels * 2, out_channels=base_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(in_channels=base_channels, out_channels=1, kernel_size=3, padding=1)
        )

    def forward(self, x, skip_connections, t_emb):
        out = self.upconv1(x)
        out = torch.cat((out, skip_connections[-1]), dim=1)
        out = self.resblock1(out, t_emb)

        out = self.upconv2(out)
        out = torch.cat((out, skip_connections[-2]), dim=1)
        out = self.resblock2(out, t_emb)

        out = torch.cat((out, skip_connections[-3]), dim=1)
        out = self.output_conv(out)
        
        return out


class Unet(nn.Module):
    def __init__(self, time_emb_dim=128, base_channels=32, max_timesteps=1000):
        super().__init__()
        # Représentation sinusoïdale des pas de temps
        pos_emb = torch.zeros((max_timesteps, time_emb_dim))
        positions = torch.arange(0, max_timesteps).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, time_emb_dim, 2).float() * (-math.log(10000.0) / time_emb_dim))
        
        pos_emb[:, 0::2] = torch.sin(positions * div_term)
        pos_emb[:, 1::2] = torch.cos(positions * div_term)
        
        self.register_buffer('pos_emb', pos_emb)
        
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, 2 * time_emb_dim),
            nn.SiLU(),
            nn.Linear(2 * time_emb_dim, time_emb_dim)
        )
        
        self.encoder = Encoder(time_emb_dim, base_channels)
        #self.bottleneck_attn = SelfAttention(base_channels * 2)
        self.decoder = Decoder(time_emb_dim, base_channels)

    def forward(self, x, t):
        t_sinusoidal = self.pos_emb[t]
        t_emb = self.time_mlp(t_sinusoidal)
        
        out, skips = self.encoder(x, t_emb)
        #out = self.bottleneck_attn(out)  # Application de l'attention globale au bottleneck
        out = self.decoder(out, skips, t_emb)
        
        return out