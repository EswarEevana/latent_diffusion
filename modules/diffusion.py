import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm

class SelfAttention(nn.Module):
    def __init__(self, channels):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.mha = nn.MultiheadAttention(embed_dim = channels, num_heads = 4)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        #print(f"In self_attention block , input shape : {x.shape}")
        b, c, h, w = x.shape
        x = x.view(b, c, h * w).swapaxes(1, 2)
        x_ln = self.ln(x).transpose(0,1)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value.transpose(0,1)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        
        return attention_value.swapaxes(2, 1).view(b, c, h, w)

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)
        
class Down(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, t):
        x = self.maxpool_conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb
    
class Up(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bicubic", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels, in_channels // 2),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, skip_x, t):
        x = self.up(x)
        #print(f"x_shape : {x.shape}")
        #print(f"skip_x shape : {skip_x.shape}")
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb
    
class UNet(nn.Module):
    def __init__(self, c_in=128, c_out=128, time_dim=256, device="cuda"):
        super().__init__()
        self.device = device
        self.time_dim = time_dim
        self.inc = DoubleConv(c_in, 64)
        self.down1 = Down(64, 128)
        self.sa1 = SelfAttention(128)
        self.down2 = Down(128, 256)
        self.sa2 = SelfAttention(256)

        self.mid1 = DoubleConv(256, 512)
        self.mid2 = DoubleConv(512, 512)
        self.mid3 = DoubleConv(512, 256)
        
        self.up2 = Up(384, 128)
        self.sa5 = SelfAttention(128)
        self.up3 = Up(192, 64)
        self.sa6 = SelfAttention(64)
        self.outc = nn.Conv2d(64, c_out, kernel_size=1)

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x, t):
        
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)

        x1 = self.inc(x)
        x2 = self.down1(x1, t)
        x2 = self.sa1(x2) #128
        x3 = self.down2(x2, t)
        x3 = self.sa2(x3) #256
        
        x3 = self.mid1(x3)
        x3 = self.mid2(x3)
        x3 = self.mid3(x3)#256
        
        x = self.up2(x3, x2, t)
        
        x = self.sa5(x)
        
        x = self.up3(x, x1, t)
        
        x = self.sa6(x)
        
        output = self.outc(x)
        
        return output

class ddpm(nn.Module):
    
    def __init__(self, timesteps = 1000,
                 beta_start = 1e-4, 
                 beta_end = 2e-2, 
                 device = 'cpu',
                 in_channels = 128,
                 out_channels = 128
                 ):
        super().__init__()
        self.num_timesteps = timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.device = device
        self.betas = self.make_beta_schedule().to(device)
        self.alphas = 1. - self.betas
        self.alpha_hat = torch.cumprod(self.alphas, dim=0)
        self.unet_model= UNet(device = self.device, c_in = in_channels, c_out = out_channels)
        #print('unet model instantiated')
    
    def make_beta_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.num_timesteps)

    def noise_schedule(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        epsilon = torch.randn_like(x)
        x_hat = sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * epsilon
        return x_hat, epsilon
    
    def sample(self, n):
        self.unet_model.eval()
        with torch.no_grad():
            x = torch.randn((n, 128, 8, 8)).to(self.device)
            for i in tqdm(reversed(range(1, self.num_timesteps)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = self.unet_model(x, t)
                alphas = self.alphas[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                betas = self.betas[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alphas) * (x - ((1 - alphas) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(betas) * noise
        self.unet_model.train()
        return x
    
    def forward(self, x):
        t = torch.randint(low=1, high=self.num_timesteps, size=(x.shape[0],))
        x_t , noise = self.noise_schedule(x, t)
        predicted_noise = self.unet_model(x_t, t)
        #print(f"predicted_noise shape : {predicted_noise.shape}")
        #print(f"x_t shape : {x_t.shape}")
        return predicted_noise, noise, x_t - predicted_noise
    
