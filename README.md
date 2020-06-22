# Exploring Texture Bias in Convolutional Neural Networks with Application to Neuroimaging Data

This is the official implementation of the experiments from Exploring Texture Bias in Convolutional Neural Networks with Application to Neuroimaging Data.


## Image Processing and Stylisation

All stylisation on 2D natural images were done using a variation of Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization which can be found in [this github repository](https://github.com/naoto0804/pytorch-AdaIN). The style images used to transfer the natural images are from [Kaggle painter by numbers dataset](https://www.kaggle.com/c/painter-by-numbers/data), which consists of paintings from various different artists.

Code used in 3D medical image processing can be found in [img_processing](img_processing/) folder.

All images were first normalised using [img_processing/normalise_images.py](img_processing/normalise_images.py) followed by one of the following image transfers.

* Gaussian filtering [img_processing/gaussian_blur.py](img_processing/gaussian_blur.py)
* Salt and Pepper filtering [img_processing/snp_filter.py](img_processing/snp_filter.py)
* Median filtering [img_processing/median_filter.py](img_processing/median_filter.py)

3D image normalisation and gaussian filtering code are variations of what Dr. Ahmed Fetit provided with, who supervised the whole project.


## Natural Image Experiments

CIFAR10 dataset was used to replicate the findings from the paper 'ImageNet-trained CNNs are Biased Towards Texture', where the methods can be found in [this github repository](https://github.com/rgeirhos/texture-vs-shape). The jupyter notebook code can be found in the folder [natural_experiment](natural_experiment/) and the seven models obtained from the experiments can be found in [natural_experiment/models](natural_experiment/models/).

Each jupyter notebook contains code used to load, normalise and split the CIFAR10 images in the format that the ResNet50 could train and test on, model architecture and model training/testing results on different images. Seven such notebooks are included in this repository which are listed below in the model name (introduced in the project report), file pairs.

* ORIG_MODEL [natural_experiment/CIFAR10 ResNet50 Model.ipynb](natural_experiment/CIFAR10%20ResNet50%20Model.ipynb)
* STY_MODEL [natural_experiment/CIFAR10 ResNet50 Model Style.ipynb](natural_experiment/CIFAR10%20ResNet50%20Model%20Style.ipynb)
* STY_RED_MODEL [natural_experiment/CIFAR10 ResNet50 Model Style Reduced.ipynb](natural_experiment/CIFAR10%20ResNet50%20Model%20Style%20Reduced.ipynb)
* ORIG_TORCH [natural_experiment/CIFAR10 ResNet50 Torch.ipynb](natural_experiment/CIFAR10%20ResNet50%20Torch.ipynb)
* STY_TORCH [natural_experiment/CIFAR10 ResNet50 Torch Style.ipynb](natural_experiment/CIFAR10%20ResNet50%20Torch%20Style.ipynb)
* STY_RED_TORCH [natural_experiment/CIFAR10 ResNet50 Torch Style Reduced.ipynb](natural_experiment/CIFAR10%20ResNet50%20Torch%20Style%20Reduced.ipynb)
* STY_BIG_TORCH [natural_experiment/CIFAR10 ResNet50 Torch Style Large.ipynb](natural_experiment/CIFAR10%20ResNet50%20Torch%20Style%20Large.ipynb)

STY_BIG_TORCH model ran for 3 weeks on a 12GB RAM GPU but still did not manage to do a full run.


## Medical Image Experiments

Publicly available datasets from [Developing Human Connectome Project](http://www.developingconnectome.org/project/) were used in the medical experiments with [deepmedic](https://github.com/deepmedic/deepmedic), a customisable 3D image segmentation model.

[medical_experiment](medical_experiment/) folder contains the model, training and testing configurations used on deepmedic as well as the testing logs of the models in this project. The testing and training configurations which can be found in folders [medical_experiment/train_config](medical_experiment/train_config/) and [medical_experiment/test_config](medical_experiment/test_config/) follow the naming convention trainConfigN and testConfigN where N represents the dataset each of the models were trained on. A list showing the model name introduced in the project report (hence the dataset the model was trained on) along with the number for N can be found below.

* N=1 MODEL_T2Norm
* N=2 MODEL_GAUS134
* N=3 MODEL_GAUS1
* N=4 MODEL_GAUS3
* N=5 MODEL_GAUS4
* N=6 MODEL_SNP010510
* N=7 MODEL_SNP01
* N=8 MODEL_SNP05
* N=9 MODEL_SNP10
* N=10 MODEL_GAUS134_SNP10
* N=11 MODEL_MEDIAN5

The test configuarations files under [medical_experiment/test_config](medical_experiment/test_config/) follow the naming convention testConfigN_dataset where dataset is the held-out test set used to test the model. Test log files under [medical_experiment/test_logs](medical_experiment/text_logs/) are named likewise.