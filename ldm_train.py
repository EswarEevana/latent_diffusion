# Importing Necessary Libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import os
import numpy as np

from tqdm import tqdm
from modules.dataset import create_dataset
from modules.diffusion_in_latent_space import ldm_pipeline
import torch.optim as optim

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(device)

#x = torch.randn(2,3,256,256)
checkpoint_path = 'models/vae_model_checkpoint_latent128.pth'
model = ldm_pipeline(device=device, vae_checkpoint = checkpoint_path, freeze_vae = True)

# Freeze VAE (encoder and decoder) parameters
# for param in model.encoder.parameters():
#     param.requires_grad = False
# for param in model.decoder.parameters():
#     param.requires_grad = False


#creating datasets
train_loader , test_loader, val_loader = create_dataset(batch_size = 32)

#Training function
def train(model, data, accumulation_steps=1, continue_training=False, checkpoint_path='models/ldm_model_checkpoint_latent128.pth'):
    # Defining optimizer
    opt = optim.AdamW(model.diffusion.parameters(), lr=1e-4)
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
            predicted_noise, noise, decoded_x = model(x)

            # Compute loss
            loss = model.loss_fn(predicted_noise, noise)
            
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
            checkpoints = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'train_loss': avg_loss,
            }
            torch.save(checkpoints, checkpoint_path)
            print(f'Model saved with best train loss: {best_loss:.4f}')

  train(model, val_loader, accumulation_steps=1, continue_training=False, checkpoint_path='models/ldm_model_checkpoint_latent128.pth')
