from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

class LinfPGDAttack:
  def __init__(self, model, epsilon, num_steps, step_size, random_start, loss_func, batch_size, use_ODI=False):
    self.model = model
    self.epsilon = epsilon
    self.num_steps = num_steps
    self.step_size = step_size
    self.rand = random_start
    self.batch_size = batch_size
    self.use_ODI = use_ODI

    if loss_func == 'xent':
      loss = model.xent
    elif loss_func == 'margin':
      label_mask = tf.one_hot(model.y_input,
                              10,
                              on_value=1.0,
                              off_value=0.0,
                              dtype=tf.float32)
      correct_logit = tf.reduce_sum(label_mask * model.pre_softmax, axis=1)
      wrong_logit = tf.reduce_max((1-label_mask) * model.pre_softmax - 1e4*label_mask, axis=1)
      loss = wrong_logit - correct_logit
      
    self.grad = tf.gradients(loss, model.x_input)[0]

    if self.use_ODI:
      self.rand_direct = tf.Variable(np.zeros((self.batch_size,10)).astype(np.float32),name='rand_direct')
      self.input_placeholder = tf.placeholder(tf.float32, shape=[self.batch_size,10])
      self.assign_op = self.rand_direct.assign(self.input_placeholder)
      loss = tf.tensordot(model.pre_softmax, self.rand_direct,axes=[[0,1],[0,1]])
      self.grad_ODI = tf.gradients(loss, model.x_input)[0]

  def lossSet(self,rand_vector,sess):
    sess.run(self.assign_op,feed_dict={self.input_placeholder: rand_vector.astype(np.float32)})

  def perturb(self, x_org, x_start, y, sess):
    if self.rand:
      x = x_org + np.random.uniform(-self.epsilon, self.epsilon, x_org.shape)
      x = np.clip(x, 0, 255) # ensure valid pixel range
    else:
      x = np.copy(x_start)

    for i in range(self.num_steps):
      if self.use_ODI:
        grad = sess.run(self.grad_ODI, feed_dict={self.model.x_input: x,
                                            self.model.y_input: y})
      else:
        grad = sess.run(self.grad, feed_dict={self.model.x_input: x,
                                            self.model.y_input: y})

      x = np.add(x, self.step_size * np.sign(grad), out=x, casting='unsafe')

      x = np.clip(x, x_org - self.epsilon, x_org + self.epsilon)
      x = np.clip(x, 0, 255) # ensure valid pixel range

    return x


