import torch
import torch.nn as nn
import math

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

        # 2*out_channels to then split in scale / shift for adaptive layer norm
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

        t_scale_shift = self.time_proj(t_emb).unsqueeze(-1).unsqueeze(-1)
        scale, shift = torch.chunk(t_scale_shift, 2, dim=1)
        
        out = self.norm2(out)
        out = out * (1 + scale) + shift     # AdaLN formula
        out = self.act2(out)
        out = self.conv2(out)

        return out + self.residual_conv(x)


class Encoder(nn.Module):
    def __init__(self, time_emb_dim, base_channels):
        super().__init__()
        self.input_conv = nn.Conv2d(in_channels=1, out_channels=base_channels, kernel_size=3, padding=1)
        self.resblock1 = ResBlock(in_channels=base_channels, out_channels=base_channels, time_emb_dim=time_emb_dim)
        self.resblock2 = ResBlock(in_channels=base_channels, out_channels=base_channels * 2, time_emb_dim=time_emb_dim)

        self.pool1 = nn.MaxPool2d(kernel_size=2, padding=0)
        self.pool2 = nn.MaxPool2d(kernel_size=2, padding=0)

    def forward(self, x, t_emb):
        skip_connections = []   # for decoder later
        
        out = self.input_conv(x)
        skip_connections.append(out)

        out = self.resblock1(out, t_emb)
        skip_connections.append(out)
        out = self.pool1(out)

        out = self.resblock2(out, t_emb)
        skip_connections.append(out)
        out = self.pool2(out)

        return out, skip_connections    # out shape : input_shape / 4, channels : base_channel*2


class Decoder(nn.Module):
    def __init__(self, time_emb_dim, base_channels):
        super().__init__()
  
        self.upconv1 = nn.ConvTranspose2d(base_channels * 2, base_channels * 2, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.resblock1 = ResBlock(in_channels=base_channels * 4, out_channels=base_channels * 2, time_emb_dim=time_emb_dim) #channels*2 because of concatenation
        
        self.upconv2 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.resblock2 = ResBlock(in_channels=base_channels * 2, out_channels=base_channels, time_emb_dim=time_emb_dim)
        
        self.output_conv = nn.Sequential(
            nn.Conv2d(in_channels=base_channels * 2, out_channels=base_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(in_channels=base_channels, out_channels=1, kernel_size=3, padding=1)
        )

    def forward(self, x, skip_connections, t_emb):
        out = self.upconv1(x)
        out = torch.cat((out, skip_connections[-1]), dim=1)     # connection with encoder
        out = self.resblock1(out, t_emb)

        out = self.upconv2(out)
        out = torch.cat((out, skip_connections[-2]), dim=1)
        out = self.resblock2(out, t_emb)

        out = torch.cat((out, skip_connections[-3]), dim=1)
        out = self.output_conv(out)
        
        return out

# attention layer at bottleneck while the size of x is low
class SelfAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.attention = nn.MultiheadAttention(embed_dim=channels, num_heads=4)
        self.norm = nn.LayerNorm((channels))

    def forward(self, x):
        b, c, h, w = x.shape
        # format (B, H * W, C) for multi head attention
        x_format = x.permute(0, 2, 3, 1).view(b, h * w, c)
        x_norm = self.norm(x_format)
        
        attn_out, _ = self.attention(x_norm, x_norm, x_norm)
        attn_out = attn_out + x_format  # residual connection
        
        return attn_out.view(b, h, w, c).permute(0, 3, 1, 2)

class Unet(nn.Module):
    def __init__(self, time_emb_dim, base_channels, time_steps):
        super().__init__()

        # sinusoidal time representation
        pos_emb = torch.zeros((time_steps, time_emb_dim))
        positions = torch.arange(0, time_steps).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, time_emb_dim, 2).float() * (-math.log(10000.0) / time_emb_dim))
        
        pos_emb[:, 0::2] = torch.sin(positions * div_term)
        pos_emb[:, 1::2] = torch.cos(positions * div_term)
        
        self.register_buffer('pos_emb', pos_emb)    # passing it later on GPU
        
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, 2 * time_emb_dim),
            nn.SiLU(),
            nn.Linear(2 * time_emb_dim, time_emb_dim)
        )

        # label embedding for finetuning
        self.label_embedding = nn.Embedding(num_embeddings=27, embedding_dim=time_emb_dim)
        
        self.encoder = Encoder(time_emb_dim, base_channels)
        self.bottleneck_attn = SelfAttention(base_channels * 2)
        self.decoder = Decoder(time_emb_dim, base_channels)

    def forward(self, x, t, labels):
        t_sinusoidal = self.pos_emb[t]
        t_emb = self.time_mlp(t_sinusoidal)
        
        t_emb = t_emb + self.label_embedding(labels)
        
        out, skips = self.encoder(x, t_emb)
        out = self.bottleneck_attn(out)
        out = self.decoder(out, skips, t_emb)
        
        return out