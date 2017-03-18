import tensorflow as tf
import summaries

def mlp(input_batch, n_hidden, lambda_=0.001, name='MLP'): 

  shape = input_batch.get_shape()
  batch_size = shape[0].value
  input_dim = shape[1].value

  with tf.variable_scope(name): 

    weights = tf.get_variable('W', (input_dim, n_hidden),
        initializer=tf.contrib.layers.xavier_initializer())

    biases = tf.get_variable('b', (n_hidden), 
        initializer=tf.constant_initializer(0.0)
        )

    hidden = tf.nn.relu(tf.matmul(input_batch, weights) + biases, name='g')
    add_loss_var(weights, lambda_)

    summaries.activation_summary(weights)
  return hidden
