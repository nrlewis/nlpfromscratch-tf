'''
The convolution.  Given an input matrix, perform a convolution 
from n_kernel kernels of shape (kernel_ht, seq_len).  
While image convolutions might move acrose both dimensions, 
sentence classification only goes in one direction across the sentence, 
learning ngram information. 

After convolution, the maximum value will be pools from each kernel's 
convution, and concatentated into one large sentence representatation

Example Input for text ( the last dimension that TF requires is removed for clarity)

For token classificaiton, the input sentence will be repeated many times, 
but the position vectors will rotate. 

All inputs will have been encoded to their embedding representation
And a position vectore will show relative position to the term to classify

[ # classify "the"
 [ 'the', 'TITLE', 'pos_0']
 [ 'quick', 'lower', 'pos_1']
 [ 'fox', 'lower', 'pos_2']
]

[ # classify "quick"
 [ 'the', 'TITLE', 'pos_-1']
 [ 'quick', 'lower', 'pos_0']
 [ 'fox', 'lower', 'pos_1']
]

[ # classify "fox"
 [ 'the', 'TITLE', 'pos_-2']
 [ 'quick', 'lower', 'pos_-1']
 [ 'fox', 'lower', 'pos_0']
]

'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from .summaries import activation_summary
from .loss import add_loss_var

def conv_max(input_batch, kernel_ht, n_kernels,  
    padding='VALID', strides=[1,1,1,1], lambda_=0.001): 
  ''' 
  The defaults are to have a kernel of shape (kernel_ht, seq_len)  convolve in 
  with valid padding and single striding.  The stride does not skip so that 
  we can learn as much ngram (or kernel_ht_gram) information as we can... Although it's possible
  to skip if one so pleases :) 
  '''

  shape = input_batch.get_shape()
  batch_size = shape[0].value
  seq_len = shape[1].value
  emb_dim = shape[2].value # kernel width
  num_channels = shape[3].value # should always be 1

  with tf.variable_scope('Convolution'): 
    kernel = tf.get_variable('kernel_%dx%d' %(kernel_ht, emb_dim), 
      (kernel_ht, emb_dim, num_channels, n_kernels),
      initializer=tf.contrib.layers.xavier_initializer())
    
    conv = tf.nn.conv2d(input_batch, kernel, strides, padding)
    bias = tf.get_variable('b', (n_kernels), 
        initializer=tf.constant_initializer(0.0)
        )
   
    add_loss_var(kernel, lambda_)
    activation_summary(kernel)
    pre_activation = tf.nn.bias_add(conv, bias)
    nonlinear_conv = tf.nn.relu(pre_activation)

    # get the maximum from the convolution.  Because the convolution
    # kernel width was the embedding_dim and the padding defaults
    # to VALID, the convolution output shape = (seq_len - kernel_ht + 1, 1) .
    # So get the max value from the full convolution sweep for each kernel
    # the pooling kernel size will be the shape of the conv output  
    # This will break if you set the padding to SAME
    pool = tf.nn.max_pool(nonlinear_conv, ksize=[1,seq_len - kernel_ht + 1, 1, 1],
                          strides=[1,1,1,1], padding='VALID', name='maxpool')
     
    # TF has some complicated striding.  Squeze the output
    sent_encoding = tf.reshape(pool, (batch_size, n_kernels))
    return sent_encoding 
  
