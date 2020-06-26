# Output Diversified Sampling (ODS)
This is the github repository for the paper "[Diversity can be Transferred: Output Diversification for White- and Black-box Attacks](https://arxiv.org/abs/2003.06878)".

## Requirement

1. Install PyTorch
1. Install pickle

## Running experiments

### ODS for Black-Box Attacks
The following experiments combine ODS with [Simple Black-Box Attack (SimBA)](https://arxiv.org/abs/1905.07121).

#### Evaluation:
The evaluation is held for 5 sample images on ImageNet (images are already resized and cropped).

* untargeted settings:
```bash
  $ python blackbox_simbaODS.py --num_sample 5 --ODS 
```

* targeted settings:
```bash
  $ python blackbox_simbaODS.py --num_sample 5 --num_step 30000 --ODS --targeted
```

### ODS for initialization of white-box attacks (ODI)
The following experiments combine ODI with PGD attack.

#### Adversarial Training:
```
python whitebox_train_cifar10.py --model-dir [PATH_TO_SAVE_FOLDER] --data-dir [PATH_TO_DATA_FOLDER]
```
#### Evaluation PGD attack with ODI:

* Evaluate PGD attack with ODI:
```bash
  $ python whitebox_pgd_attack_cifar10_ODI.py --ODI-num-steps 2 --model-path [PATH_TO_THE_MODEL] --data-dir [PATH_TO_DATA_FOLDER]
```

* Evaluate PGD attack with naive random initialization (sampled from a uniform distribution):
```bash
  $ python whitebox_pgd_attack_cifar10_ODI.py --ODI-num-steps 0 --model-path [PATH_TO_THE_MODEL] --data-dir [PATH_TO_DATA_FOLDER]
```

## Acknowledgement
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
