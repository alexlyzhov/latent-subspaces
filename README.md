# latent-subspaces
Implementation of ["Learning latent subspaces in variational autoencoders"](http://papers.nips.cc/paper/7880-learning-latent-subspaces-in-variational-autoencoders.pdf) (NIPS'18).

`csvae_toy/` contains scripts with CSVAE model definition and training procedure on toy data (swiss roll from sklearn).

`condvae_celeba/` contains scripts for training VAE and Conditional VAE on CelebA dataset, the folder `model/` with architectures of VAE and Conditional VAE,  the folder `weights/` with the best weights and training ditails and trainig history dataframe.

`attr_classifier/` contains of the notebook for training and testing attributes classification model (ResNet34) and the weights for this model.

`results/` contains of the notebooks with models performance results. There you can find the generation examples, computed accuracy of generation and the exmples of style transfer.



Contributors: Artem Shafarostov, Marina Pominova, Alexander Lyzhov, Elizaveta Lazareva
