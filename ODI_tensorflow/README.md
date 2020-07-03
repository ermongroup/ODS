# Output Diversified Sampling for Initialization (ODI)

The code is for ODI in tensorflow.

## Running experiments

### Evaluation PGD attack with ODI (on CIFAR-10):

* Download pre-trained model (trained by [MadryLab](https://github.com/MadryLab/cifar10_challenge))
```bash
  $ python fetch_model.py secret
```

* Evaluate PGD attack with ODI (ODI-PGD):
```bash
  $ python eval_ODI_PGD.py --num_restart 20 --num_step_ODI 2 --data_path [PATH_TO_CIFAR10]
```

* Evaluate PGD attack with naive random initialization (sampled from a uniform distribution):
```bash
  $ python eval_ODI_PGD.py --num_restart 20 --num_step_ODI 0 --data_path [PATH_TO_CIFAR10]
```

## Acknowledgement
Our codes for PGD are based on [CIFAR10 Adversarial Examples Challenge](https://github.com/MadryLab/cifar10_challenge). 

