# Output Diversified Sampling (ODS)
This is the github repository for the paper "[Diversity can be Transferred: Output Diversification for White- and Black-box Attacks](https://arxiv.org/abs/2003.06878)".

## Requirement

Please install PyTorch, pickle, argparse, and numpy

## Running experiments

### ODS for score-based black-box attacks
The following experiments combine ODS with [Simple Black-Box Attack (SimBA)](https://arxiv.org/abs/1905.07121).

#### Evaluation:
The evaluation is held for 5 sample images on ImageNet (images are already resized and cropped).

```shell
# untargeted settings with ODS:
python blackbox_simbaODS.py --num_sample 5 --ODS 
# targeted settings with ODS:
python blackbox_simbaODS.py --num_sample 5 --num_step 30000 --ODS --targeted
```

___

### ODS for decision-based black-box attacks
The following experiments combine ODS with [Boundary Attack](https://arxiv.org/abs/1712.04248).

#### Additional Requirement

Please install Foolbox, Python>=3.6

#### Evaluation:
The evaluation is held for 5 sample images on ImageNet (images are already resized and cropped).

```shell
# untargeted settings with ODS:
python blackbox_boundaryODS.py --num_sample 5 --ODS 
# targeted settings with ODS:
python blackbox_boundaryODS.py --num_sample 5 --ODS --targeted
# untargeted settings with random sampling:
python blackbox_boundaryODS.py --num_sample 5 
# targeted settings with random sampling:
python blackbox_boundaryODS.py --num_sample 5 --targeted
```

#### Acknowledgement
Our codes for Boundary Attack are based on [Foolbox repo](https://github.com/bethgelab/foolbox).

___

### ODS for initialization of white-box attacks (ODI)
The following experiments combine ODI with PGD attack.

#### Training of target model (Adversarial Training):
```shell
python whitebox_train_cifar10.py --model-dir [PATH_TO_SAVE_FOLDER] --data-dir [PATH_TO_DATA_FOLDER]
```

#### Evaluation PGD attack with ODI:

```shell
# Evaluate PGD attack with ODI:
python whitebox_pgd_attack_cifar10_ODI.py --ODI-num-steps 2 --model-path [PATH_TO_THE_MODEL] --data-dir [PATH_TO_DATA_FOLDER] 
# Evaluate PGD attack with naive random initialization (sampled from a uniform distribution):
python whitebox_pgd_attack_cifar10_ODI.py --ODI-num-steps 0 --model-path [PATH_TO_THE_MODEL] --data-dir [PATH_TO_DATA_FOLDER]
```

#### Acknowledgement
Our codes for white-box attacks are based on [TRADES official repo](https://github.com/yaodongyu/TRADES).

## Citation
If you use this code for your research, please cite our paper:

```
@article{tashiro2020ods,
  title={Diversity can be Transferred: Output Diversification for White- and Black-box Attacks},
  author={Tashiro, Yusuke and Song, Yang and Ermon, Stefano},
  journal={arXiv preprint arXiv:2003.06878},
  year={2020}
}
```
