import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter


def train(model, EPOCHS, train_dataset, train_loss, optimizer,device):
    # Set device to GPU if available, else CPU
    
    # Initialize TensorBoard writer for logging losses
    writer = SummaryWriter()
    
    # Iterate through each epoch
    for epoch in range(EPOCHS):
        print("EPOCHS : ", epoch+1, "/", EPOCHS)
        
        # Initialize cumulative loss variables for tracking per epoch
        sum_loss = 0
        sum_KLD = 0
        sum_recon = 0
        
        # Loop over each batch in the training dataset
        for i, (imag, _) in enumerate(train_dataset):
            
            # Reset cumulative loss for each batch
            sum_loss = 0
            sum_KLD = 0
            sum_recon = 0
            
            # Set the model to training mode
            model.train(True)
            
            # Zero the gradients from the optimizer for each batch
            optimizer.zero_grad()
            
            # Move images to the specified device (CPU or GPU)
            image = imag.to(device)
            
            # Forward pass: get outputs and latent variables from the model
            outputs, z, z_mean, z_logvar = model(image)
            outputs, z, z_mean, z_logvar = (
                outputs.to(device),
                z.to(device),
                z_mean.to(device),
                z_logvar.to(device),
            )
            
            # Calculate loss using the provided loss function
            loss = train_loss(image, outputs, z, z_mean, z_logvar)
            
            # Clean up to free memory for further processing
            del image, outputs, z, z_mean, z_logvar
            
            # Backward pass: compute gradients
            loss.backward()
            
            # Update model parameters based on gradients
            optimizer.step()
            
            # Accumulate total loss for the batch
            sum_loss += loss.detach()
            
            # Get the separate KLD and reconstruction losses for logging
            KLD, recon = train_loss.get_split_losses()
            sum_KLD += KLD.detach()
            sum_recon += recon.detach()
            
            # Delete loss tensor to save memory
            del loss
        
        # Print KLD and reconstruction losses for each epoch
        print(sum_KLD)
        print(sum_recon)
        
        # Log KLD and reconstruction losses in TensorBoard
        writer.add_scalar('KLD/ith_batch_seen', sum_KLD, epoch)
        writer.add_scalar('recon/ith_batch_seen', sum_recon, epoch)
        
        # Log overall loss for each epoch in TensorBoard
        writer.add_scalar('Loss/per_epoch', sum_loss, epoch)
        
        # Print average loss per sample for the epoch
        print(sum_loss / len(train_dataset))
