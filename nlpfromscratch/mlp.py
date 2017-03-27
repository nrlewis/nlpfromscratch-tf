import tensorflow as tf

from .loss import add_loss_var
from .summaries import activation_summary 

def linear(input_batch, n_hidden, name, lambda_=0.001): 

  shape = input_batch.get_shape()
  batch_size = shape[0].value
  input_dim = shape[1].value

  with tf.variable_scope(name): 

    weights = tf.get_variable('W', (input_dim, n_hidden),
        initializer=tf.contrib.layers.xavier_initializer())

    biases = tf.get_variable('b', (n_hidden), 
        initializer=tf.constant_initializer(0.0)
        )

    add_loss_var(weights, lambda_)
    activation_summary(weights)
    linear = tf.nn.bias_add(tf.matmul(input_batch, weights),  biases)

  return linear 


