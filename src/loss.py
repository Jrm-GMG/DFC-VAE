import torchvision
import torch
from torch import nn

class FeaturedConceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Load VGG19 model's feature extractor layers, up to a certain layer
        features = torchvision.models.vgg19(torchvision.models.VGG19_Weights.IMAGENET1K_V1).features
        self.vgg = nn.Sequential(*list(features.children())[:-25])  # Retain layers until specific depth
        
        # Disable gradient computation for the VGG layers to save memory and computation
        self.vgg.requires_grad_(False)
        
        # Define the preprocessing transformations for VGG19 model input
        self.process = torchvision.models.VGG19_Weights.IMAGENET1K_V1.transforms()
        
        # Dictionary to store activations for specific layers
        self.act = {}
        
    def _getActivation(self, name):
        # Hook function to capture the output of specified layers
        def hook(model, input, output):
            self.act[name] = output  # Store the layer output in the activations dictionary
        return hook

    def forward(self, output, target):
        # Register hooks to capture activations at specific layers for `output`
        l1 = self.vgg[3].register_forward_hook(self._getActivation('relu1_2'))
        l2 = self.vgg[6].register_forward_hook(self._getActivation('relu2_1'))
        l3 = self.vgg[11].register_forward_hook(self._getActivation('relu3_1'))

        # Pass `output` through VGG19 and capture activations
        self.vgg(self.process(output))        
        output_l1 = self.act['relu1_2']
        output_l2 = self.act['relu2_1']
        output_l3 = self.act['relu3_1']

        # Pass `target` through VGG19 and capture activations
        self.vgg(self.process(target))        
        target_l1 = self.act['relu1_2']
        target_l2 = self.act['relu2_1']
        target_l3 = self.act['relu3_1']

        # Calculate loss as the mean squared error between corresponding feature maps
        L = torch.mean(torch.square(output_l1 - target_l1)) + \
            torch.mean(torch.square(output_l2 - target_l2)) + \
            torch.mean(torch.square(output_l3 - target_l3))

        # Remove hooks after forward pass to free memory
        l1.remove()
        l2.remove()
        l3.remove()
        
        # Return average loss from selected feature layers
        return L / 3.0


class FPL_Loss(nn.Module):
    def __init__(self, epochs, total_im, beta=1.e-4, alpha=1.e+3):
        super().__init__()
        
        # Initialize memory variables to store KLD and reconstruction losses
        self.KLD_mem = 0
        self.recon_mem = 0
        
        # Coefficients for reconstruction loss and KLD loss
        self.beta = beta
        self.alpha = alpha
        
        # Initialize the featured conceptual loss (reconstruction) component
        self.recon = FeaturedConceptualLoss()
        
    def forward(self, x, z_x, z, z_mean, z_logvar):
        # Compute reconstruction loss using featured conceptual loss between `z_x` and `x`
        reconstruction = self.recon(z_x, x)
        
        # Calculate Kullback-Leibler Divergence (KLD) for latent distribution regularization
        KLD = -0.5 * torch.mean(torch.sum(1 - torch.exp(z_logvar) - z_mean * z_mean + z_logvar, dim=1))
        
        # Store KLD and reconstruction losses in memory variables
        self.KLD_mem = KLD
        self.recon_mem = reconstruction
        
        # Return total loss as a weighted sum of reconstruction and KLD losses
        return (self.beta * reconstruction + self.alpha * KLD)

    def get_split_losses(self):
        # Return individual losses with respective weightings for analysis or logging
        return self.alpha * self.KLD_mem, self.beta * self.recon_mem
