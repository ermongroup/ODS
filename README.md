# Output Diversified Initialization
This is the github repository for the paper "Output Diversified Initialization for Adversarial Attacks".

## Running experiments

### Adversarial Training:

```
python train_cifar10.py
```
### Evaluation PGD attack with Output Diversified Initialization (ODI):

* Evaluate PGD attack with ODI:
```bash
  $ python odi_pgd_attack_cifar10.py --ODI-num-steps 2 --model-path [PATH_TO_THE_MODEL]
```

* Evaluate PGD attack with naive random initialization (sampled from a uniform distribution):
```bash
  $ python odi_pgd_attack_cifar10.py --ODI-num-steps 0 --model-path [PATH_TO_THE_MODEL]
```
## Acknowledgement
Our codes for training and PGD are based on [TRADES official repo](https://github.com/yaodongyu/TRADES).

