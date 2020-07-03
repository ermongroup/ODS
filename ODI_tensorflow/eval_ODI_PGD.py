from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import sys
import pickle

import tensorflow as tf
import numpy as np 

import cifar10_input
from model import Model
from pgd_attack import LinfPGDAttack

import argparse

parser = argparse.ArgumentParser(description='CIFAR PGD Attack Evaluation')
parser.add_argument('--save_folder', type=str,default='',
                    help='name of save folder')
parser.add_argument('--data_path', type=str,default='../data/cifar10/',
                    help='path of data folder')
parser.add_argument('--model_path', type=str,default='./models/model_0/checkpoint-70000',
                    help='path of model folder')    
parser.add_argument('--evalSize',
                    default=10000,type=int,
                    help='number of evaluated images')
parser.add_argument('--eval_batch_size',
                    default=100,type=int,
                    help='batch size')
parser.add_argument('--num_restart',
                    default=20,type=int,
                    help='number of restarts')
parser.add_argument('--rand_start',
                    default=1,type=int,
                    help='random initialization for attack')
parser.add_argument('--eps',
                    default=8.0,type=float,
                    help='size of l_inf ball')
parser.add_argument('--step_size',
                    default=2.0,type=float,
                    help='step size for PGD')
parser.add_argument('--num_step',
                    default=20,type=int,
                    help='number of PGD step')
parser.add_argument('--step_size_ODI',
                    default=8.0,type=float,
                    help='step size for ODI')
parser.add_argument('--num_step_ODI',
                    default=2,type=int,
                    help='number of ODI step')                      
args = parser.parse_args()

cifar = cifar10_input.CIFAR10Data(args.data_path)

model = Model(mode='eval')
var_all = tf.get_collection(tf.GraphKeys.VARIABLES)
saver = tf.train.Saver(var_all)

if args.num_step_ODI > 0:
  init_ODI = LinfPGDAttack(model,
                        args.eps,
                        args.num_step_ODI,
                        args.step_size_ODI,
                        args.rand_start,
                        'margin', 
                        args.eval_batch_size,
                        use_ODI=True)
attack_PGD = LinfPGDAttack(model,
                      args.eps,
                      args.num_step,
                      args.step_size,
                      False,
                      'margin',
                      args.eval_batch_size)
          
          
def evaluate_checkpoint(sess):
  # Iterate over the samples batch-by-batch
  num_batches = int(math.ceil(args.evalSize / args.eval_batch_size))
  correct_list = np.ones(args.evalSize)

  for ibatch in range(num_batches):
    bstart = ibatch * args.eval_batch_size
    bend = min(bstart + args.eval_batch_size, args.evalSize)

    x_batch = cifar.eval_data.xs[bstart:bend]
    y_batch = cifar.eval_data.ys[bstart:bend]
    x_batch_org = x_batch
    
    if args.num_step_ODI > 0:
      ran_ = np.random.uniform(-1.0,1.0, (args.eval_batch_size,10))
      init_ODI.lossSet(ran_,sess)
      x_batch = init_ODI.perturb(x_batch_org,x_batch, y_batch, sess)
    elif args.rand_start == 1:
      x_batch = x_batch_org + (np.random.uniform(-args.eps, args.eps, x_batch_org.shape) ) 
      x_batch= np.clip(x_batch, 0., 255.) # ensure valid pixel range

    x_batch = attack_PGD.perturb(x_batch_org, x_batch, y_batch, sess)

    dict_adv = {model.x_input: x_batch,
                model.y_input: y_batch}

    cur_isCorrect, = sess.run([model.correct_prediction],
                                    feed_dict = dict_adv)
    correct_list[bstart:bend] = cur_isCorrect              

  return correct_list
 
#main
sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver.restore(sess, args.model_path)

model_dir = 'results/'+args.save_folder
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
is_correct = np.ones(args.evalSize)
acc_curve = np.zeros(args.num_restart)
for i in range(args.num_restart):
    curr_correct = evaluate_checkpoint(sess)
    is_correct = curr_correct * is_correct
    acc_curve[i] = is_correct.mean()
    with open(model_dir+'/result.pk', 'wb') as f:
        pickle.dump([acc_curve,is_correct], f)
print(acc_curve)
        
