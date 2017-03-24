import tensorflow as tf
import numpy as np
from nlpfromscratch.convnet import conv_max

class ConvTest(tf.test.TestCase): 
  
  def test_inits(self): 
    
    batch_size = 4 
    seq_len = 5
    emb_dim = 10  
    num_channels = 1
    kernel_ht = 2 
    n_kernels = 10

    input_ = np.random.uniform(0,1, 
        (batch_size, seq_len, emb_dim, num_channels))

    input_batch = tf.placeholder(tf.float32, 
        shape=(batch_size, seq_len, emb_dim, num_channels))

    conv = conv_max(input_batch, kernel_ht, n_kernels)

    with self.test_session() as sess: 
      sess.run(tf.global_variables_initializer())
      encoded_sent = sess.run(conv, feed_dict={input_batch:input_})
      self.assertAllEqual((batch_size, n_kernels), encoded_sent.shape) 
      losses = tf.get_collection('losses')
      self.assertEqual(len(losses), 1) 


if __name__ == '__main__': 
  tf.test.main()
      
