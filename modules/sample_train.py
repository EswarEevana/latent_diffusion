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
from dataset import create_dataset
from vae_model import VAE

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

# vae = VAE()
# x = torch.randn(2,3,256,256)
# y,me,var=vae(x)
# print(f"input shape : {x.shape}")
# print(f"mean and variance shape : {var.shape}")
# print(f"output shape : {y.shape}")

train_loader,_,_ = create_dataset(batch_size=32)

model = VAE().to(device)

def train(model, data, accumulation_steps=8, continue_training=False, checkpoint_path='models/vae_model_checkpoint_latent128.pth'):
    # Define optimizer
    opt = torch.optim.Adam(model.parameters(), lr=1e-5)
    start_epoch = 0
    best_loss = float('inf')  # Initialize with infinity to track the best loss

    # Learning rate scheduler: Reduce LR on plateau
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=5, verbose=True, min_lr=1e-7)

    # Load checkpoint if training is to be resumed
    if continue_training and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        opt.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        best_val_loss = checkpoint['train_loss']
        print(f"Resuming training from epoch {start_epoch} with best validation loss {best_val_loss:.4f}")
    
    num_epochs = 50  # Number of epochs to train AFTER loading the checkpoint
    total_epochs = start_epoch + num_epochs

    # Training loop
    print("Training started!!!!!!!!!")
    for epoch in range(start_epoch, total_epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0
        opt.zero_grad()  # Zero gradients at the start of the epoch

        for i, batch in enumerate(data):
            x = batch['pixel_values']
            x = x.to(device)  # Move to GPU
            
            # Forward pass
            recon_x, z_mean1, z_log_var1 = model(x)

            # Compute loss
            loss = model.loss_fn(recon_x, x, z_mean1, z_log_var1)
            
            # Accumulate gradients (scaled by accumulation steps)
            loss = loss / accumulation_steps
            loss.backward()

            # Print loss for every few iterations
            if i % 10 == 0:  # Print every 10th iteration
                print(f"Epoch: [{epoch+1}/{total_epochs}] Iteration: {i+1} --> Loss: {loss.item():.4f}")

            # Update weights and zero gradients after 'accumulation_steps' batches
            if (i + 1) % accumulation_steps == 0:
                opt.step()  # Perform optimizer step
                opt.zero_grad()  # Clear gradients for the next set of accumulation steps

            # Accumulate running loss
            running_loss += loss.item() * accumulation_steps  # Re-scale to original loss

        # Calculate average loss for the epoch
        avg_loss = running_loss / len(data)
        print(f"Epoch: [{epoch+1}/{total_epochs}] --> Average Loss: {avg_loss:.4f}")

        # Update learning rate based on validation loss
        scheduler.step(avg_loss)  # Reduce LR if there's no improvement

        # Save checkpoint if validation loss improves
        if avg_loss < best_loss:
            best_loss = avg_loss
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'train_loss': avg_loss,
            }
            torch.save(checkpoint, checkpoint_path)
            print(f'Model saved with best train loss: {best_loss:.4f}')

    return model
train(model, train_loader, accumulation_steps=8, continue_training=True, checkpoint_path='models/vae_model_checkpoint_latent128.pth')