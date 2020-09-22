# Rethinking the Value of Labels for Improving Class-Imbalanced Learning

This is the source code implementation of ``Rethinking the Value of Labels for Improving Class-Imbalanced Learning''.

We provide the code for both the semi-supervised & self-supervised imbalanced learning implementation.
We plan to clean up all the code and release them soon.


## Dependencies
1. PyTorch (>=1.2, tested on 1.4)
2. yaml
3. scikit-learn
4. TensorboardX


## Prerequisites & Installation
1. Download CIFAR & SVHN dataset, and place them in your <data_path>
2. Download ImageNet & iNaturalist 2018 dataset, and place them in your <data_path>
3. Download .txt files of train/val/test splits for ImageNet-LT & iNaturalist 2018 from [2], and place it in `./dataset` and `./imagenet_inat/data/`


## Main Files
- ``train.py'': train model with (without) SSP, on CIFAR-LT / SVHN-LT
- ``train_semi.py'': train model with extra unlabeled data, on CIFAR-LT / SVHN-LT
- ``imagenet_inat/main.py'': train model with (without) SSP, on ImageNet-LT / iNaturalist 2018


## Main Arguments
- ``--dataset'': name of chosen long-tailed dataset
- ``--imb_factor'': imbalance factor (inverse value of the defined "imbalance ratio \rho" in our paper)
- ``--imb_factor_unlabel'': imbalance factor for unlabeled data
- ``--pretrained_model'': path to (self-supervised) pre-trained models

Other arguments are listed in each file with explanations.


## Usage

### Semi-supervised learning with pseudo-labeling
Generate pseudo-labels using base classifier trained on original imbalanced dataset
```
python gen_pseudolabels.py --data_dir <data_path> --output_dir <output_path> --output_filename <save_name>
```

Train with unlabeled data on CIFAR-10-LT with \rho=100, and \rho_U=100
```
python train_semi.py --dataset cifar10 --imb_factor 0.01 --imb_factor_unlabel 0.01
```

### Self-supervised Pre-training
Rotation SSP on CIFAR-10-LT with \rho=100
```
python pretrain_rot.py --dataset cifar10 --imb_factor 0.01
```

MoCo SSP on ImageNet-LT
```
python pretrain_moco.py --dataset imagenet --data <data_path>
```

### Network training with SSP models
Train on CIFAR-10-LT with \rho=100
```
python train_semi.py --dataset cifar10 --imb_factor 0.01 --pretrained_model <path_to_your_model>
```

Train on ImageNet-LT / iNat
```
python imagenet_inat/main.py --cfg <path_to_your_config>
```


## Acknowledgements
This code is partly based on the open-source implementations from [1,2,3,4].


## Reference
[1] https://github.com/zhmiao/OpenLongTailRecognition-OLTR.\
[2] https://github.com/facebookresearch/classifier-balancing.\
[3] https://github.com/kaidic/LDAM-DRW.\
[4] https://github.com/facebookresearch/moco.
