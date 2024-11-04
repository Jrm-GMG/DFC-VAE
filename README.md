# DFC-VAE
Implementation of a Variational Autoencoder to generate faces. This implementation is based on the following paper: https://arxiv.org/abs/1610.00291. A brief report detailing the training process and personal observations is in progress.

Face reconstruction:

On the left is the original image, and on the right is the image reconstructed by the VAE.

<div>
	<img src='/images/recon_one.png'>
  <img src='/images/recon_two.png'>
  <img src='/images/recon_three.png'>
  <img src='/images/recon_four.png'>
</div>

Faces generation:

We can generate new faces by sampling a vector from the prior distribution of the latent space and passing it through the decoder

<div>
  <img src='/images/generation0.png'>
	<img src='/images/generation1.png'>
</div>

Adding smile to faces:

With labeled attributes, we can identify specific feature vectors in the latent space. For example, to create a 'smiling' vector, we compute the average vector of smiling images and subtract the average vector of non-smiling images. Adding this resulting vector to the latent representation of a non-smiling face and passing it through the decoder enables us to generate an image of the same face, now with a smile.

<div>
  <img src='/images/one.png'>
  <img src='/images/two.png'>
  <img src='/images/three.png'>
  <img src='/images/four.png'>
</div>
