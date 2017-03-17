import tensorflow as tf


def mlp(input_batch, n_hidden, n_classes, lambda_=0.001): 

  shape = input_batch.get_shape().value
  batch_size = shape[0]
  input_dim = shape[1]

  with tf.variable_scope('MLP'): 

    weights = tf.get_variable('W', (input_dim, n_hidden),
        initializer=tf.contrib.layers.xavier_initializer())


    biases = tf.get_variable('b', (n_hidden), 
        initializer=tf.constant_initializer(0.0)
        )

    hidden = tf.nn.relu(tf.nn.matmul(input_batch, weights) + biases, name='g')

    # save the regularized weight in a collection
    weight_decay = tf.multiply(tf.nn.l2_loss(weights), lambda_, name='h1_reg')
    tf.add_to_collection('losses', weight_decay)

  return hidden
