# Viral Predictor for mRNA Evolution (VPRE)
A deep-learning and statistical analysis framework applied to model and predict mutation trajectories of SARS-CoV-2 spike proteins for supporting pre-emptive vaccine design. 

## Introduction
VPRE is implemented in Jupyter notebooks that use machine learning algorithms in Keras and PyMC3.

Keras is used for training and using variational autoencoders, which constructs a continuous latent embedding of SARS-CoV2 spike protein sequence space. After training, we can generate new sequences from the latent space.

PyMC3 is used for gaussian process regression, which allows us to model the trajectory of a value across time and extrapolate the trend into the future, thus providing a way to forecast the spike protein's evolution from the latent space. We can then decode the numbers into spike proteins to predict its future conformation.  

For more information, see http://virosight.ca and email us at ubcigem@gmail.com. 

We also call for people who are interested in applying deep learning in modeling biological sequences to try the framework of VPRE for different scenarios. 

## Requirements
* Tensorflow
* Keras
* PyMC3
* Biopython
* Scipy
* Pandas
