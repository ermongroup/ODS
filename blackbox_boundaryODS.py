import attacksODS
from foolbox.models import PyTorchModel
from foolbox.criteria import Misclassification, TargetedMisclassification
import torch
import torchvision.models as models
import numpy as np
import argparse
import pickle

parser = argparse.ArgumentParser()

parser.add_argument('--device', default='cuda:0', help='Device for Attack')
parser.add_argument('--data_folder', type=str, default='./data', help='target model to use')
parser.add_argument('--model_name', type=str, default='Resnet50', help='target model to use')
parser.add_argument('--smodel_name', type=str, default='', help='surrogate model to use. Blank means multi surrogate models')
parser.add_argument('--targeted', action='store_true', help='perform targeted attack')
parser.add_argument('--ODS', action='store_true', help='perform ODS')
parser.add_argument('--num_step', type=int, default=10000, help='maximum step size of Boundary attack')
parser.add_argument('--num_sample', default=5,type=int)
args = parser.parse_args()

device = torch.device(args.device if torch.cuda.is_available() else "cpu")

model_list = ['Resnet34','Resnet50', 'VGG19','Densenet121','Mobilenet']
attr_list = ['resnet34','resnet50','vgg19_bn','densenet121','mobilenet_v2']
preprocessing = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], axis=-3)

for i in range(len(model_list)):
    if model_list[i] == args.model_name:
        pretrained_model = getattr(models,attr_list[i])(pretrained=True).eval()
        fmodel = PyTorchModel(pretrained_model, bounds=(0, 1), preprocessing=preprocessing)

surrogate_model_list = []
if args.smodel_name == '':#multiSurrogate
    for i in range(len(model_list)):
        if model_list[i] != args.model_name:
            pretrained_model = getattr(models,attr_list[i])(pretrained=True).eval()
            surrogate_model_list.append( PyTorchModel(pretrained_model, bounds=(0, 1), preprocessing=preprocessing) )
else:
    for i in range(len(model_list)):
        if model_list[i] == args.smodel_name:
            pretrained_model = getattr(models,attr_list[i])(pretrained=True).eval()
            surrogate_model_list.append( PyTorchModel(pretrained_model, bounds=(0, 1), preprocessing=preprocessing) )


#dataload 
url_main = args.data_folder + '/imagenet_5sample.pk'
url_tgt = args.data_folder + '/imagenet_5sample_target.pk'
with open(url_main, 'rb') as f:  
    images_all,labels_all = pickle.load(f)

with open(url_tgt, 'rb') as f:  
    images_tgt,labels_tgt = pickle.load(f)
    
distList_finalstep = np.zeros(args.num_sample)
distListAll = np.zeros((args.num_sample,(int)(args.num_step/100)+1))
for i in range(args.num_sample):
    images = images_all[i:i+1].to(device)
    labels = labels_all[i:i+1].to(device)
        
    if args.targeted:
        imgTarget = images_tgt[i:i+1].to(device)
        classVec = labels_tgt[i:i+1].to(device)
        criterion = TargetedMisclassification(classVec)
        attack = attacksODS.BoundaryAttack(
            tensorboard=False,steps=args.num_step,surrogate_models=surrogate_model_list,ODS=args.ODS)
        advs = attack.run(fmodel, images, criterion,starting_points=imgTarget)
        history = attack.normHistory
    else:
        criterion = Misclassification(labels)
        attack = attacksODS.BoundaryAttack(init_attack=None,
            tensorboard=False,steps=args.num_step,surrogate_models=surrogate_model_list,ODS=args.ODS)
        advs = attack.run(fmodel, images, criterion)
        history = attack.normHistory

    print('image %d: query %d, current dist = %.4f' % (
                    i + 1, args.num_step, (advs[0]-images[0]).norm()))

    distList_finalstep[i] = (advs[0]-images[0]).norm()
    distListAll[i] = history
    prefix = '_targeted' if args.targeted else ''
    #with open('res_boundary' +prefix+'.pk', 'wb') as f:  # Python 3: open(..., 'wb')
    #    pickle.dump([i+1,distList_finalstep,distListAll], f)


