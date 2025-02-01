# Importing Necessary Libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import os
import numpy as np
#from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import numpy as np
from tqdm import tqdm


#Encoder
class Encoder(nn.Module):
    def __init__(self, input_channels=3):
      super(Encoder, self).__init__()
      #self.encoder_features = {}


      self.layer1 = nn.Conv2d(in_channels = input_channels, out_channels = 64, stride = 1, kernel_size = 3, padding = 1)
      self.layer2 = nn.Conv2d(in_channels = 64, out_channels = 64, stride = 2, kernel_size = 3, padding = 1)
      self.layer3 = nn.Conv2d(in_channels = 64, out_channels = 64, stride = 1, kernel_size = 3, padding = 1)
      self.actv1  = nn.ReLU()
      self.layer4 = nn.Conv2d(in_channels = 64, out_channels = 64, stride = 1, kernel_size = 3, padding = 1)
      #self.add1   = self.layer4 + self.layer2
      self.actv2 = nn.ReLU()

      self.layer5 = nn.Conv2d(in_channels = 64, out_channels = 128, stride = 2, kernel_size = 3, padding = 1)
      self.layer6 = nn.Conv2d(in_channels = 128, out_channels = 128, stride = 1, kernel_size = 3, padding = 1)
      self.actv3 = nn.ReLU()
      self.layer7 = nn.Conv2d(in_channels = 128, out_channels = 128, stride = 1, kernel_size = 3, padding = 1)
      #self.add2  = self.layer7 + self.layer5
      self.actv4 = nn.ReLU()

      self.layer8 = nn.Conv2d(in_channels = 128, out_channels = 256, stride = 2, kernel_size = 3, padding = 1)
      self.layer9 = nn.Conv2d(in_channels = 256, out_channels = 256, stride = 1, kernel_size = 3, padding = 1)
      self.actv5 = nn.ReLU()
      self.layer10 = nn.Conv2d(in_channels = 256, out_channels = 256, stride = 1, kernel_size = 3, padding = 1)
      #self.add3 = self.layer10 + self.layer8
      self.actv6 = nn.ReLU()

      self.layer11 = nn.Conv2d(in_channels = 256, out_channels = 512, stride = 2, kernel_size = 3, padding = 1)
      self.layer12 = nn.Conv2d(in_channels = 512, out_channels = 512, stride = 1, kernel_size = 3, padding = 1)
      self.actv7 = nn.ReLU()
      self.layer13 = nn.Conv2d(in_channels = 512, out_channels = 512, stride = 1, kernel_size = 3, padding = 1)
      #self.add4 = self.layer13 + self.layer11
      self.actv8 = nn.ReLU()

      self.layer14 = nn.Conv2d(in_channels =512, out_channels = 512, stride = 2, kernel_size = 3, padding = 1)
      self.layer15 = nn.Conv2d(in_channels = 512, out_channels = 512, stride = 1, kernel_size = 3, padding = 1)
      self.actv9 = nn.ReLU()
      self.layer16 = nn.Conv2d(in_channels = 512, out_channels = 512, stride = 1, kernel_size = 3, padding = 1)
      #self.add5 = self.layer16 + self.layer14
      self.actv10 = nn.ReLU()

      self.layer17 = nn.Conv2d(in_channels = 512, out_channels = 512, stride = 1, kernel_size = 3, padding = 1)
      self.actv11 = nn.ReLU()

      #self.concatenate1 = torch.hstack((self.actv11, self.layer14))

      self.layer18 = nn.Conv2d(in_channels = 1024, out_channels = 512, stride = 1, kernel_size = 3, padding = 1)
      self.actv12 = nn.ReLU()
      self.layer19 = nn.Conv2d(in_channels = 512, out_channels = 512, stride = 1, kernel_size = 3, padding = 1)
      #self.add6   = self.layer19 + self.activ11
      self.actv13 = nn.ReLU()
      #self.concatenate2 = torch.hstack((self.actv13, self.layer14))

      self.layer20 = nn.Conv2d(in_channels = 1024, out_channels = 512, stride = 1, kernel_size = 3, padding = 1)
      self.actv14 = nn.ReLU()

      self.mean  = nn.Conv2d(in_channels = 512, out_channels = 128, stride = 1, kernel_size = 3, padding = 1)
      self.logvar = nn.Conv2d(in_channels = 512, out_channels = 128, stride = 1, kernel_size = 3, padding = 1)

    def forward(self, x):
      #print(f"shape of input : {x.shape}")
      x = self.layer1(x)
      x = self.layer2(x)
      #self.encoder_features['layer2'] = x
      res1 = x
      x = self.layer3(x)
      x = self.actv1(x)
      x = self.layer4(x)

      #x = self.add1(x)
      x = self.actv2(x + res1)
      #print(f"shape of input before layer5 : {x.shape}")
      x= self.layer5(x)
      #self.encoder_features['layer5'] = x
      res2 = x
      x = self.layer6(x)
      x = self.actv3(x)
      x = self.layer7(x)

      #x = self.add2(x)
      x = self.actv4(x + res2)
      #print(f"shape of input before layer8 : {x.shape}")
      x = self.layer8(x)
      #self.encoder_features['layer8'] = x
      res3 = x
      x = self.layer9(x)
      x = self.actv5(x)
      x = self.layer10(x)

      #x = self.add3(x)
      x = self.actv6(x + res3)
      #print(f"shape of input before layer11 : {x.shape}")
      x = self.layer11(x)
      #self.encoder_features['layer11'] = x
      res4 = x
      x = self.layer12(x)
      x = self.actv7(x)
      x = self.layer13(x)

      #x = self.add4(x)
      x = self.actv8(x + res4)
      #print(f"shape of input before layer14 : {x.shape}")
      x = self.layer14(x)
      res5 = x
      x = self.layer15(x)
      x = self.actv9(x)
      x = self.layer16(x)

      #x = self.add5(x)
      x = self.actv10(x + res5)
      #print(f"shape of input before layer17 : {x.shape}")
      x = self.layer17(x)
      res6 = x
      x = self.actv11(x)
      #x = self.concatenate1(x)
      x = torch.cat((x,res5), dim = 1)
      x = self.layer18(x)
      x = self.actv12(x)
      x = self.layer19(x)

      #x = self.add6(x)
      x = self.actv13(x+res6)
      #print(f"shape of input before layer20 : {x.shape}")
      #x = self.concatenate2(x)
      x = torch.cat((x, res5), dim = 1)
      x = self.actv14(self.layer20(x))
      #print(f"shape of input after forward pass layer 20 : {x.shape}")

      #x = torch.flatten(x, start_dim = 1)
      #print(f"shape of input after flatten : {x.shape}")
      mean = self.mean(x)
      logvar = self.logvar(x)
      #print(f"shape of mean : {mean.shape}")
      #print(f"shape of logvar : {logvar.shape}")

      return mean, logvar #, self.encoder_features

#Decoder
class Decoder(nn.Module):
    def __init__(self, output_channels=3):
      super(Decoder, self).__init__()
      #self.fc = nn.Linear(latent_dim, 512*16)
      # shape = [B,512,4,4]
      self.translayer1 = nn.ConvTranspose2d(in_channels = 128, out_channels = 512, stride = 2, kernel_size = 3, padding = 1, output_padding  = 1) # shape = [B,512,8,8]
      self.layer1 = nn.Conv2d(in_channels = 512, out_channels = 512, stride = 1, kernel_size = 3, padding = 1)
      self.actv1 = nn.ReLU()
      #concatenation of actv1 and encoder layer11
      self.layer2 = nn.Conv2d(in_channels = 512, out_channels = 512, stride = 1, kernel_size = 3, padding = 1)
      self.actv2 = nn.ReLU()
      self.layer3 = nn.Conv2d(in_channels = 512, out_channels = 512, stride = 1, kernel_size = 3, padding = 1)
      # addition of layer3 and actv1
      self.actv3 = nn.ReLU()
      #concatenation of actv3 and encoder layer11

      self.translayer2 = nn.ConvTranspose2d(in_channels = 512, out_channels = 256, stride = 2, kernel_size = 3, padding = 1, output_padding =1) # shape = [B,256,16,16]
      self.layer4 = nn.Conv2d(in_channels = 256, out_channels = 256, stride = 1, kernel_size = 3, padding = 1)
      self.actv4 = nn.ReLU()
      #concatenation of actv4 and encoder layer8
      self.layer5 = nn.Conv2d(in_channels = 256, out_channels = 256, stride = 1, kernel_size = 3, padding = 1)
      self.actv5 = nn.ReLU()
      self.layer6 = nn.Conv2d(in_channels = 256, out_channels = 256, stride = 1, kernel_size = 3, padding = 1)
      # addition of layer6 and actv4
      self.actv6 = nn.ReLU()
      #concatenation of actv6 and encoder layer8

      self.translayer3 = nn.ConvTranspose2d(in_channels = 256, out_channels = 128, stride = 2, kernel_size = 3, padding = 1, output_padding  = 1) # shape = [B,128,32,32]
      self.layer7 = nn.Conv2d(in_channels = 128, out_channels = 128, stride = 1, kernel_size = 3, padding = 1)
      self.actv7 = nn.ReLU()
      #concatenation of actv7 and encoder layer5
      self.layer8 = nn.Conv2d(in_channels = 128, out_channels = 128, stride = 1, kernel_size = 3, padding = 1)
      self.actv8 = nn.ReLU()
      self.layer9 = nn.Conv2d(in_channels = 128, out_channels = 128, stride = 1, kernel_size = 3, padding = 1)
      # addition of layer9 and actv7
      self.actv9 = nn.ReLU()
      #concatenation of actv9 and encoder layer5
     
      self.translayer4 = nn.ConvTranspose2d(in_channels = 128, out_channels = 64, stride = 2, kernel_size = 3, padding = 1, output_padding  = 1) # shape = [B,64,64,64]
      self.layer10 = nn.Conv2d(in_channels = 64, out_channels = 64, stride = 1, kernel_size = 3, padding = 1)
      self.actv10 = nn.ReLU()
      #concatenation of actv10 and encoder layer2
      self.layer11 = nn.Conv2d(in_channels = 64, out_channels = 64, stride = 1, kernel_size = 3, padding = 1)
      self.actv11 = nn.ReLU()
      self.layer12 = nn.Conv2d(in_channels = 64, out_channels = 64, stride = 1, kernel_size = 3, padding = 1)
      # addition of layer12 and actv10
      self.actv12 = nn.ReLU()
      #concatenation of actv12 and encoder layer2

      self.translayer5 = nn.ConvTranspose2d(in_channels = 64, out_channels = 64, stride = 2, kernel_size = 3, padding = 1, output_padding  = 1) # shape = [B,64,128,128]
      self.layer13 = nn.Conv2d(in_channels = 64, out_channels = 64, stride = 1, kernel_size = 3, padding = 1)
      self.actv13 = nn.ReLU()
      self.layer14 = nn.Conv2d(in_channels = 64, out_channels = output_channels, stride = 1, kernel_size = 3, padding = 1)

    def forward(self, x):
      #x = self.fc(x)
      #x = x.view(-1,512,4,4) #torch.reshape(x, (-1, 512, 4, 4))

      x = self.translayer1(x)
      x = self.actv1(self.layer1(x))
      res1 = x
      #concatenation of x with encoder layer11
      #x = torch.cat((x, features1['layer11'],features2['layer11']), dim = 1)
      x = self.actv2(self.layer2(x))
      x = self.layer3(x)
      x = self.actv3(x + res1)

      #concatenation of x with encoder layer11
      #x = torch.cat((x, features1['layer11'], features2['layer11']), dim = 1)
      x = self.translayer2(x)
      x = self.actv4(self.layer4(x))
      res2 = x

      #concatenation of x with encoder layer8
      #x = torch.cat((x, features1['layer8'], features2['layer8']), dim = 1)
      x = self.actv5(self.layer5(x))
      x = self.layer6(x)
      x = self.actv6(x + res2)

      #concatenation of x with encoder layer8
      #x = torch.cat((x, features1['layer8'], features2['layer8']), dim = 1)
      x = self.translayer3(x)
      x = self.actv7(self.layer7(x))
      res3 = x

      #concatenation of x with encoder layer5
      #x = torch.cat((x, features1['layer5'], features2['layer5']), dim = 1)
      x = self.actv8(self.layer8(x))
      x = self.layer9(x)
      x = self.actv9(x + res3)

      #concatenation of x with encoder layer5
      #x = torch.cat((x, features1['layer5'], features2['layer5']), dim = 1)
      x = self.translayer4(x)
      x = self.actv10(self.layer10(x))
      res4 = x

      #concatenation of x with encoder layer2
      #x = torch.cat((x, features1['layer2'], features2['layer2']), dim = 1)
      x = self.actv11(self.layer11(x))
      x = self.layer12(x)
      x = self.actv12(x + res4)

      #concatenation of x with encoder layer2
      #x = torch.cat((x, features1['layer2'], features2['layer2']), dim = 1)
      x = self.translayer5(x)
      x = self.actv13(self.layer13(x))
      x = self.layer14(x)

      return x


class VAE(nn.Module):
    def __init__(self, in_channels=3):
        super(VAE,self).__init__()
        self.encoder = Encoder(in_channels)
        self.decoder = Decoder(output_channels = in_channels)
        
    def reparametrize(self, z_mean, z_logvar):
        std = torch.exp(0.5*z_logvar) # standard deviation
        eps = torch.randn_like(std) # `randn_like` as we need the same size
        sample = z_mean + (eps * std) # sampling
        return sample

    def forward(self,x1):
        
        z_mean1, z_logvar1 = self.encoder(x1)

        z = self.reparametrize(z_mean1, z_logvar1)
        
        #print(f"dimension of the input before decoder : {z.shape}")
        
        x = self.decoder(z)
        #print(f"dimension of the input after decoder : {x.shape}")
        return x, z_mean1, z_logvar1
  
    def loss_fn(self,recon_x, x, z_mean1, z_logvar1):


        # Reconstruction loss
        #BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
        l1 = F.l1_loss(recon_x,x, reduction='sum')

        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2) KL divergence loss given by Kingma and Welling in Auto-Encoding Variational Bayes. ICLR, 2014
        # variance = sigma^2
        KLD1 = -0.5 * torch.sum(1 + z_logvar1 - z_mean1**2 - torch.exp(z_logvar1))
        # KLD2 = -0.5 * torch.sum(1 + z_logvar2 - z_mean2**2 - torch.exp(z_logvar2))
        return  KLD1 + l1
    
