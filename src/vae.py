import torchvision
import torch
from torch import nn
from collections import OrderedDict

# Get cpu, gpu or mps device for training.

class VAE(nn.Module):
    '''
    VAE model class
    x: input image
    z: latent variables
    y: reconstruction of the input
    '''
    
    def __init__(self, input_dimension, latent_dimension=100):
        super().__init__()
        # Initialize general parameters for VAE
        self.sigmoid = nn.Sigmoid()
        self.latent_dimension = latent_dimension
        self.input_dimension = input_dimension
        self.build()  # Builds the model architecture

    def build(self):
        ###################
        #### Inference ####
        ###################
        # Encoder network from input `x` to latent variable `z`
        dic_layers = OrderedDict()
        
        # Initial convolution layer to reshape the input
        dic_layers['Conv_reshape'] = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=4, stride=(2, 2), padding=1, padding_mode='replicate')
        dic_layers['normalization_reshape'] = nn.BatchNorm2d(3)
        dic_layers['activation_reshape'] = nn.LeakyReLU()
        
        # Encoder Conv layers with increasing channels
        dic_layers['Conv0'] = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=4, stride=(2, 2), padding=1, padding_mode='replicate')
        dic_layers['normalization0'] = nn.BatchNorm2d(32)
        dic_layers['activation0'] = nn.LeakyReLU()
        
        dic_layers['Conv1'] = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=(2, 2), padding=1, padding_mode='replicate')
        dic_layers['normalization1'] = nn.BatchNorm2d(64)
        dic_layers['activation1'] = nn.LeakyReLU()

        dic_layers['Conv2'] = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=(2, 2), padding=1, padding_mode='replicate')
        dic_layers['normalization2'] = nn.BatchNorm2d(128)
        dic_layers['activation2'] = nn.LeakyReLU()

        dic_layers['Conv3'] = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=(2, 2), padding=1, padding_mode='replicate')
        dic_layers['normalization3'] = nn.BatchNorm2d(256)
        dic_layers['activation3'] = nn.LeakyReLU()
        
        # Flatten the final convolution output for linear layer input
        dic_layers['Flatten'] = nn.Flatten()
        
        # Sequential model representing the encoder pipeline from x to z
        self.mlp_x_z = nn.Sequential(dic_layers)
        
        # Linear layers for mean and log variance of latent variable z
        self.z_mean = nn.Linear(256 * 4 * 4, self.latent_dimension)
        self.z_logvar = nn.Linear(256 * 4 * 4, self.latent_dimension)

        ######################
        #### Generation x ####
        ######################
        # Decoder network to reconstruct `x` from `z`
        dic_layers = OrderedDict()
        
        # Fully connected layer to expand latent space to feature map size
        dic_layers['FC0'] = nn.Linear(self.latent_dimension, 256 * 4 * 4)
        dic_layers['invFlatten'] = nn.Unflatten(1, (256, 4, 4))

        # Decoder layers with upsampling and transposed convolutions
        dic_layers['UP0'] = torch.nn.Upsample(scale_factor=2)
        dic_layers['TConv0'] = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1, padding_mode='replicate')
        dic_layers['normalization0'] = nn.BatchNorm2d(128)
        dic_layers['activation0'] = nn.LeakyReLU()

        dic_layers['UP1'] = torch.nn.Upsample(scale_factor=2)
        dic_layers['TConv1'] = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1, padding_mode='replicate')
        dic_layers['normalization1'] = nn.BatchNorm2d(64)
        dic_layers['activation1'] = nn.LeakyReLU()

        dic_layers['UP2'] = torch.nn.Upsample(scale_factor=2)
        dic_layers['TConv2'] = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1, padding_mode='replicate')
        dic_layers['normalization2'] = nn.BatchNorm2d(32)
        dic_layers['activation2'] = nn.LeakyReLU()

        dic_layers['UP3'] = torch.nn.Upsample(scale_factor=2)
        dic_layers['TConv3'] = nn.Conv2d(in_channels=32, out_channels=3, kernel_size=3, padding=1, padding_mode='replicate')
        dic_layers['normalization3'] = nn.BatchNorm2d(3)
        dic_layers['activation3'] = nn.LeakyReLU()
        
        dic_layers['UP4'] = torch.nn.Upsample(scale_factor=2)
        dic_layers['TConv_up'] = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, padding=1, padding_mode='replicate')
        
        # Sequential model representing the decoder pipeline from z to reconstructed x
        self.mlp_z_x = nn.Sequential(dic_layers)
        
        # Sigmoid activation function for output layer to constrain output values
        self.s = nn.Sigmoid()

    def reparameterization(self, mean, logvar):
        # Reparameterization trick to sample from a Gaussian distribution
        std = torch.exp(0.5 * logvar)  # Calculate the standard deviation
        eps = torch.randn_like(std)    # Random tensor with same shape as std
        z = torch.addcmul(mean, std, eps)  # Compute sampled z with mean and std deviation
        return z

    def encode(self, x):
        # Encodes input `x` into latent variables `z`, `z_mean`, and `z_logvar`
        x_z = self.mlp_x_z(x)           # Pass input through encoder pipeline
        z_mean = self.z_mean(x_z)       # Compute mean of z
        z_logvar = self.z_logvar(x_z)   # Compute log variance of z
        z = self.reparameterization(z_mean, z_logvar)  # Sample z using reparameterization
        return z, z_mean, z_logvar

    def decode(self, z):
        # Decodes latent variable `z` back to original data space `x`
        z_x = self.mlp_z_x(z)           # Pass latent variable through decoder
        return self.s(z_x)              # Apply sigmoid to output for normalization

    def forward(self, x):
        # Defines forward pass: encodes and then decodes
        z, z_mean, z_logvar = self.encode(x)  # Encode input to latent variables
        z_x = self.decode(z)                  # Decode latent variable to reconstruct x
        return z_x, z, z_mean, z_logvar       # Return reconstructed x and latent details
