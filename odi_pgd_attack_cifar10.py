from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable
import torch.optim as optim
from torchvision import datasets, transforms
from models.wideresnet import *
from models.resnet import *
import numpy as np

parser = argparse.ArgumentParser(description='PyTorch CIFAR PGD Attack Evaluation with Output Diversified Initialization')
parser.add_argument('--test-batch-size', type=int, default=100, metavar='N',
                    help='input batch size for testing (default: 100)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--num-restarts', default=20, type=int,
                    help='number of restarts')
parser.add_argument('--epsilon', default=8/255, type=float,
                    help='perturbation')
parser.add_argument('--num-steps', default=20, type=int,
                    help='perturb number of steps')
parser.add_argument('--step-size', default=2/255, type=float,
                    help='perturb step size')
parser.add_argument('--ODI-num-steps', default=2, type=int,
                    help='ODI perturb number of steps')
parser.add_argument('--ODI-step-size', default=8/255, type=float,
                    help='ODI perturb step size')
parser.add_argument('--lossFunc', help='loss function for PGD',
                    type=str, default='margin', choices=['xent','margin'])
parser.add_argument('--random',
                    default=True,
                    help='random initialization for PGD')
parser.add_argument('--model-path',
                    default='./checkpoints/model-latest.pt',
                    help='model for white-box attack evaluation')
parser.add_argument('--source-model-path',
                    default='./checkpoints/model-latest.pt',
                    help='source model for black-box attack evaluation')
parser.add_argument('--target-model-path',
                    default='./checkpoints/model-latest.pt',
                    help='target model for black-box attack evaluation')
parser.add_argument('--white-box-attack', default=1,type=int,
                    help='whether perform white-box attack')
parser.add_argument('--arch', help='architectures',
                    type=str, default='ResNet', choices=['ResNet','WideResNet'])
parser.add_argument('--archTarget', help='architectures of target model',
                    type=str, default='ResNet', choices=['ResNet','WideResNet'])
parser.add_argument('--archSource', help='architectures of source model',
                    type=str, default='ResNet', choices=['ResNet','WideResNet'])

args = parser.parse_args()
print(args)

# settings
use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

# set up data loader
transform_test = transforms.Compose([transforms.ToTensor(),])
testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)

def margin_loss(logits,y):

    logit_org = logits.gather(1,y.view(-1,1))
    logit_target = logits.gather(1,(logits - torch.eye(10)[y].to("cuda") * 9999).argmax(1, keepdim=True))
    loss = -logit_org + logit_target
    loss = torch.sum(loss)
    return loss

def _pgd_whitebox(model,
                  X,
                  y,
                  epsilon=args.epsilon,
                  num_steps=args.num_steps,
                  step_size=args.step_size,
                  ODI_num_steps=args.ODI_num_steps,
                  ODI_step_size=args.ODI_step_size
                  ):
    out = model(X)
    acc_clean = (out.data.max(1)[1] == y.data).float().sum()
    X_pgd = Variable(X.data, requires_grad=True)

    randVector_ = torch.FloatTensor(*model(X_pgd).shape).uniform_(-1.,1.).to(device)
    if args.random:
        random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).to(device)
        X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

    for i in range(ODI_num_steps + num_steps):
        opt = optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()
        with torch.enable_grad():
            if i < ODI_num_steps:
                loss = (model(X_pgd) * randVector_).sum()
            elif args.lossFunc == 'xent':
                loss = nn.CrossEntropyLoss()(model(X_pgd), y)
            else:
                loss = margin_loss(model(X_pgd),y)
        loss.backward()
        if i < ODI_num_steps: 
            eta = ODI_step_size * X_pgd.grad.data.sign()
        else:
            eta = step_size * X_pgd.grad.data.sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
        X_pgd = Variable(X.data + eta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)

    acc_each = (model(X_pgd).data.max(1)[1] == y.data).detach().cpu().numpy() 
    acc_pgd = (model(X_pgd).data.max(1)[1] == y.data).float().sum()
    return acc_clean, acc_pgd, acc_each


def _pgd_blackbox(model_target,
                  model_source,
                  X,
                  y,
                  epsilon=args.epsilon,
                  num_steps=args.num_steps,
                  step_size=args.step_size,
                  ODI_num_steps=args.ODI_num_steps,
                  ODI_step_size=args.ODI_step_size
                  ):
    out = model_target(X)
    acc_clean = (out.data.max(1)[1] == y.data).float().sum()
    X_pgd = Variable(X.data, requires_grad=True)
    randVector_ = torch.FloatTensor(*out.shape).uniform_(-1.,1.).to(device)
    if args.random:
        random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).to(device)
        X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

    for i in range(ODI_num_steps + num_steps):
        opt = optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()
        with torch.enable_grad():
            if i < ODI_num_steps:
                loss = (model_source(X_pgd) * randVector_).sum()
            elif args.lossFunc == 'xent':
                loss = nn.CrossEntropyLoss()(model_source(X_pgd), y)
            else:
                loss = margin_loss(model_source(X_pgd),y)
        loss.backward()
        if i < ODI_num_steps: 
            eta = ODI_step_size * X_pgd.grad.data.sign()
        else:
            eta = step_size * X_pgd.grad.data.sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
        X_pgd = Variable(X.data + eta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)

    acc_each = (model_target(X_pgd).data.max(1)[1] == y.data).detach().cpu().numpy() 
    acc_pgd = (model_target(X_pgd).data.max(1)[1] == y.data).float().sum()
    return acc_clean, acc_pgd, acc_each


def eval_adv_test_whitebox(model, device, test_loader):
    """
    evaluate model by white-box attack
    """
    model.eval()

    acc_total = np.ones(10000)
    acc_curve = []
    for _ in range(args.num_restarts):
        natural_acc_oneshot = 0
        robust_acc_oneshot = 0
        for i, [data, target] in enumerate(test_loader):
            bstart = i * args.test_batch_size
            bend = (i+1) * args.test_batch_size
            data, target = data.to(device), target.to(device)
            # pgd attack
            X, y = Variable(data, requires_grad=True), Variable(target)
            acc_natural, acc_robust, acc_each = _pgd_whitebox(model, X, y)
            acc_total[bstart:bend] = acc_total[bstart:bend] * acc_each
            natural_acc_oneshot += acc_natural
            robust_acc_oneshot += acc_robust
        print('natural_acc_oneshot: ', natural_acc_oneshot)
        print('robust_acc_oneshot: ', robust_acc_oneshot)
        print('accuracy_total: ', acc_total.sum())
        acc_curve.append(acc_total.sum())
        print('accuracy_curve: ', acc_curve)


def eval_adv_test_blackbox(model_target, model_source, device, test_loader):
    """
    evaluate model by black-box attack
    """
    model_target.eval()
    model_source.eval()

    acc_total = np.ones(10000)
    acc_curve = []
    for _ in range(args.num_restarts):
        natural_acc_oneshot = 0
        robust_acc_oneshot = 0
        for i, [data, target] in enumerate(test_loader):
            bstart = i * args.test_batch_size
            bend = (i+1) * args.test_batch_size
            data, target = data.to(device), target.to(device)
            # pgd attack
            X, y = Variable(data, requires_grad=True), Variable(target)
            acc_natural, acc_robust,acc_each = _pgd_blackbox(model_target, model_source, X, y)
            acc_total[bstart:bend] = acc_total[bstart:bend] * acc_each
            natural_acc_oneshot += acc_natural
            robust_acc_oneshot += acc_robust
        print('natural_acc_oneshot: ', natural_acc_oneshot)
        print('robust_acc_oneshot: ', robust_acc_oneshot)
        print('accuracy_total: ', acc_total.sum())
        acc_curve.append(acc_total.sum())
        print('accuracy_curve: ', acc_curve)


def main():

    if args.white_box_attack:
        # white-box attack
        print('pgd white-box attack')

        model = ResNet18().to(device) if args.arch=='ResNet' else WideResNet().to(device)
        model.load_state_dict(torch.load(args.model_path))

        eval_adv_test_whitebox(model, device, test_loader)
    else:
        # black-box attack
        print('pgd black-box attack')
        model_target = ResNet18().to(device) if args.archTarget=='ResNet' else WideResNet().to(device)
        model_target.load_state_dict(torch.load(args.target_model_path))
        
        model_source = ResNet18().to(device) if args.archSource=='ResNet' else WideResNet().to(device)
        model_source.load_state_dict(torch.load(args.source_model_path))

        eval_adv_test_blackbox(model_target, model_source, device, test_loader)

if __name__ == '__main__':
    main()
