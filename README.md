# latent-subspaces
Implementation of ["Learning latent subspaces in variational autoencoders"](http://papers.nips.cc/paper/7880-learning-latent-subspaces-in-variational-autoencoders.pdf) (NIPS'18).

CSVAE is an autoencoder model based on plain VAE and is an implementation of the idea of mutual information minimization between a feature and a part of latent space to constraint the structure of latent mapping for richer representations.

We implement Conditional subspace VAE (CSVAE) architecture, architectures of corresponding competing approaches (CondVAE, CondVAE-Info, VAE), and do a benchmarking in a number of setups including reconstruction quality, latent space structuring capability and attribute transfer.

[Project report is here](report.pdf). Light CSVAE models on toy data with epoch-by-epoch visualization are [here](https://yadi.sk/d/Fdc8uPq3yO-lSQ).

`csvae_toy/` contains scripts with CSVAE model definition and training procedure on toy data along with visualizations for easier interpretation and quantitative results. `definitions.py` contains definitions for a number of models including CSVAE, `train.py` is a parametrized script for easy training, `csvae_toy.ipynb` is for playing around with representations and fast propotyting. Other notebooks are for CelebA training and evaluation.

`condvae_celeba/` contains scripts for training VAE and Conditional VAE on CelebA dataset, the folder `model/` with architectures of VAE and Conditional VAE, and the folder `weights/` with the best weights, training details and dataframe with training log.

`cvae_info/scripts` contains script with for training cvae_info on celebA dataset. [Article](https://arxiv.org/pdf/1711.05175.pdf)

`attr_classifier/` contains the notebook for training and testing the attribute classification model (ResNet34) and the weights for this model.

`results/` contains notebooks with CelebA benchmarking results. There you can find the generation examples and metrics and examples of style transfer.


Contributors: Artem Shafarostov, Marina Pominova, Alexander Lyzhov, Elizaveta Lazareva
