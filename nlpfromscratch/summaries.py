import tensorflow as tf
def activation_summary(W): 
  ''' shamelessly stolen from CIFAR 10 in TF tutorial  :) ''' 

  tf.summary.histogram(W.op.name + '/activations', W)
  tf.summary.scalar(W.op.name + '/sparsity', tf.nn.zero_fraction(W))

def metrics(labels): 

  with tf.variable_scope('metrics'): 

    for metric in ['precision', 'recall', 'f1']:
      with tf.variable_scope(metric): 
        for label in labels: 
          m = tf.get_variable(label, 1, initializer=tf.constant_initializer(0.0))
          


