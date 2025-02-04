import torch
import torch.nn as nn
from diffusion import ddpm
#from dataset import create_dataset
from vae_model import VAE
import torch.nn.functional as F


class ldm_pipeline(nn.Module):
    def __init__(self, device = None,
                vae_checkpoint = None,
                freeze_vae = False ):
        super().__init__()
        self.device = device
        self.vae = VAE()
        
        if vae_checkpoint:
            print(f"Loading pretrained VAE from: {vae_checkpoint}")
            checkpoint = torch.load(vae_checkpoint, map_location=device)
            self.vae.load_state_dict(checkpoint['model_state_dict'])

        if freeze_vae:
            for param in self.vae.parameters():
                param.requires_grad = False
        
        self.encoder = self.vae.encoder
        
        self.reparametrize = self.vae.reparametrize
        
        self.diffusion = ddpm(in_channels = 128, out_channels = 128, device = self.device, timesteps = 1000)
        #print("diffusion model instantiated")
         
        self.decoder = self.vae.decoder
        
    def forward(self, x):
        
        z_mean, z_logvar = self.encoder(x)
        latent_x = self.reparametrize(z_mean,z_logvar)
        #print(f"latent shape : {latent_x.shape}")
        predicted_noise, noise, diffused_image = self.diffusion(latent_x)
        decoded_x = self.decoder(diffused_image)
        
        return predicted_noise, noise, decoded_x
    
    def loss_fn(self, noise, predicted_noise):
        
        return F.mse_loss(noise, predicted_noise)
    
    def sampling(self, n=8):
        latent = self.diffusion.sample(n)
        x = self.decoder(latent)
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        return x
    
