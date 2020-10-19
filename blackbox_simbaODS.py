import argparse
import os
import pickle

import torch
import torchvision.models as models
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--device', default='cuda:0', help='Device for Attack')
parser.add_argument('--data_folder', type=str, default='./data', help='target model to use')
parser.add_argument('--model_name', type=str, default='Resnet50', help='target model to use')
parser.add_argument('--smodel_name', type=str, default='', help='surrogate model to use. Blank means multi surrogate models')
parser.add_argument('--targeted', action='store_true', help='perform targeted attack')
parser.add_argument('--ODS', action='store_true', help='perform ODS')
parser.add_argument('--num_step', type=int, default=10000, help='maximum step size of Boundary attack')
parser.add_argument('--num_sample', default=10,type=int, help='number of image samples')
parser.add_argument('--step_size', default=0.2,type=float, help='step size per iteration')

args = parser.parse_args()

class Normalize(torch.nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.mean = mean
        self.std = std

    def forward(self, input):
        size = input.size()
        x = input.clone()
        for i in range(size[1]):
            x[:,i] = (x[:,i] - self.mean[i])/self.std[i]
        return x
        
def margin_loss(logits,y):
    logit_org = logits.gather(1,y.view(-1,1))
    logit_target = logits.gather(1,(logits - torch.eye(1000)[y].to(logits.device) * 9999).argmax(1, keepdim=True))
    loss = -logit_org + logit_target
    loss = torch.sum(loss)
    return loss

device = torch.device(args.device if torch.cuda.is_available() else "cpu")
mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]

model_list = ['Resnet34','Resnet50', 'VGG19','Densenet121','Mobilenet']
attr_list = ['resnet34','resnet50','vgg19_bn','densenet121','mobilenet_v2']

for i in range(len(model_list)):
    if model_list[i] == args.model_name:
        pretrained_model = getattr(models,attr_list[i])(pretrained=True)
model = torch.nn.Sequential(
    Normalize(mean, std),
    pretrained_model
)
model.to(device).eval()

surrogateModelList = []
if args.smodel_name == "": #multi surrogate models
    for i in range(len(model_list)):
        if args.model_name != model_list[i]:
            pretrained_model = getattr(models,attr_list[i])(pretrained=True)
            pretrained_model = torch.nn.Sequential(
                Normalize(mean, std),
                pretrained_model
            )
            surrogateModelList.append(pretrained_model.to(device).eval())
else: #single surrogate model
    for i in range(len(model_list)):
        if args.smodel_name == model_list[i]:
            pretrained_model = getattr(models,attr_list[i])(pretrained=True)
            pretrained_model = torch.nn.Sequential(
                Normalize(mean, std),
                pretrained_model
            )
            surrogateModelList.append(pretrained_model.to(device).eval())


url_main = args.data_folder + '/imagenet_5sample.pk'
url_tgt = args.data_folder + '/imagenet_5sample_target.pk'
with open(url_main, 'rb') as f:  
    images_all,labels_all = pickle.load(f)
with open(url_tgt, 'rb') as f:  
    images_tgt,labels_tgt = pickle.load(f)

loss_func = torch.nn.functional.cross_entropy if args.targeted else margin_loss

distList = np.zeros(args.num_sample)
qList = np.zeros(args.num_sample)
succList = np.zeros(args.num_sample)
for i in range(args.num_sample):
    images = images_all[i:i+1].to(device)
    labels = labels_all[i:i+1].to(device)
    labels_attacked = labels.clone()
    if args.targeted:
        labels_attacked[0] = labels_tgt[i].item()
    logits = model(images).data
    correct = (torch.argmax(logits, dim=1) != labels_attacked) if args.targeted else (torch.argmax(logits, dim=1) == labels_attacked)

    if correct:
        X_best = images.clone()
        loss_best = loss_func( logits.data,labels_attacked) * (-1 if args.targeted else 1)
        nQuery = 1 # query for the original image
        for m in range(args.num_step):
            if args.ODS:
                X_grad = torch.autograd.Variable(X_best.data, requires_grad=True)
                random_direction = torch.rand((1,1000)).to(device) * 2 - 1
                ind = np.random.randint(len(surrogateModelList))
                with torch.enable_grad(): 
                    loss = (surrogateModelList[ind](X_grad) * random_direction).sum()
                loss.backward()
                delta = X_grad.grad.data / X_grad.grad.norm()
            else:
                ind1 = np.random.randint(3)
                ind2 = np.random.randint(224)
                ind3 = np.random.randint(224)
                delta = torch.zeros(X_best.shape).cuda()
                delta[0,ind1,ind2,ind3] = 1
            for sign in [1,-1]:
                X_new = X_best + args.step_size * sign * delta
                X_new = torch.clamp(X_new,0,1)
                logits = model(X_new).data
                nQuery+= 1
                loss_new = loss_func(logits.data,labels_attacked) * (-1 if args.targeted else 1)
                if loss_best<loss_new:
                    X_best= X_new
                    loss_best = loss_new
                    break


            success = (torch.argmax(logits, dim=1) == labels_attacked) if args.targeted else (torch.argmax(logits, dim=1) != labels_attacked)
            if success:
                distList[i] =  (X_best-images).norm()
                qList[i] =  nQuery
                succList[i] = 1
                print('image %d: attack is successful. query = %d, dist = %.4f' % (
                        i + 1, nQuery, (X_best-images).norm() ) )
                break
            if m == args.num_step - 1:
                print('image %d: attack is not successful (query = %d)' % (
                        i + 1, nQuery ) )
    else:
        print('image %d: already adversary' % (i + 1))
        
    prefix = '_targeted' if args.targeted else ''
    print('image %d: average dist=%.4f, average query=%.4f ' % (
            i+1,distList[:i+1].mean(), qList[:i+1].mean() ) )        
