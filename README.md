# DFC-VAE
Implementation of a Variational Autoencoder to generate faces. This implementation is based on the following paper: https://arxiv.org/abs/1610.00291. 
The goal of this paper is to introduce the use of a pre-trained deep convolutional neural network to compute the reconstruction loss. We utilize its hidden features to define a perceptual loss for VAE training.

A brief report detailing the training process and personal observations is in progress.

# Face reconstruction:

On the left is the original image, and on the right is the image reconstructed by the VAE. Similarly to the VAE-123 model from the paper, we see that the VAE focuses on the images while the background is blurred.

<div>
 <img src='/images/recon_one.png'>
  <img src='/images/recon_two.png'>
  <img src='/images/recon_three.png'>
  <img src='/images/recon_four.png'>
</div>

# Faces generation:

We can generate new faces by sampling a vector from the prior distribution of the latent space and passing it through the decoder

<div>
  <img src='/images/generation0.png'>
	<img src='/images/generation1.png'>
</div>

# Adding smile to faces:

With labeled attributes, we can identify specific feature vectors in the latent space. For example, to create a 'smiling' vector, we compute the average vector of smiling images and subtract the average vector of non-smiling images. Adding this resulting vector to the latent representation of a non-smiling face and passing it through the decoder enables us to generate an image of the same face, now with a smile.

The VAE not only adds a smile to the mouth but also alters the overall expression of the face. The raised cheekbones and the changes in the angle of the eyes play a significant role in this transformation!

<div>
  <img src='/images/one.png'>
<img src='/images/four.png'>
  <img src='/images/two.png'>
  <img src='/images/three.png'>
  
</div>
