from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf

def add_loss_var(W, lambda_): 
  
  # save the regularized weight in a collection
  weight_decay = tf.multiply(tf.nn.l2_loss(W), lambda_, name=W.op.name + '_loss')
  tf.add_to_collection('losses', weight_decay)

def reg_softmax_loss(logits, labels): 

  with tf.name_scope('loss'): 
    cross_ent = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, 
        labels=labels)
    avg_loss = tf.reduce_mean(cross_ent)
    reg_loss = tf.add_n(tf.get_collection('losses'))
    total_loss = tf.add(avg_loss, reg_loss)

  tf.summary.scalar('loss', total_loss)
  return total_loss
