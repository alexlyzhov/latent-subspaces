# latent-subspaces
Implementation of ["Learning latent subspaces in variational autoencoders"](http://papers.nips.cc/paper/7880-learning-latent-subspaces-in-variational-autoencoders.pdf) (NIPS'18).

`csvae_toy/` contains scripts with CSVAE model definition and training procedure on toy data.

`condvae_celeba/` contains scripts for training VAE and Conditional VAE on CelebA dataset, the folder `model/` with architectures of VAE and Conditional VAE, and the folder `weights/` with the best weights, training details and dataframe with training log.

`cvae_info/scripts` contains script with for training cvae_info on celebA dataset

`attr_classifier/` contains the notebook for training and testing the attribute classification model (ResNet34) and the weights for this model.

`results/` contains notebooks with CelebA benchmarking results. There you can find the generation examples and metrics and examples of style transfer.


Contributors: Artem Shafarostov, Marina Pominova, Alexander Lyzhov, Elizaveta Lazareva
