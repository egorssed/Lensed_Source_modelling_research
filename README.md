# Lensed-source-modelling-with-autoencoders-and-differentiable-programming

Neural networks are more and more employed in strong lens modelling. 
In this project, the goal is to train a variational autoencoder (VAE) to learn the surface brightness of typical lensed galaxies. 
A potential improvement might be introduced by using the recently established « disentangled VAE » architecture. 
With such networks, the goal would be to extract specific features from the source galaxy, such as size, orientation and ellipticity, 
directly from the abstract space defined by the VAE. Furthermore, using the differentiable programming framework JAX, 
the VAE could be included in a larger modelling pipeline in a modular way.


