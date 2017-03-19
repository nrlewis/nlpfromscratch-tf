import tensorflow as tf
import numpy as np
from nlpfromscratch.mlp import linear

class MLPTest(tf.test.TestCase): 
  
  def test_inits(self): 
    
    batch_size = 4
    input_dim = 5
    input_ = np.array([[i/10.] * input_dim for i in range(batch_size)], 
              dtype=np.float32)
    n_hidden = 10

    input_batch = tf.placeholder(tf.float32, shape=(batch_size, input_dim))
    wx_plus_b = linear(input_batch, n_hidden, 'test')

    with self.test_session() as sess: 
      sess.run(tf.global_variables_initializer())
      logits = sess.run(wx_plus_b, feed_dict={input_batch:input_})
      self.assertAllEqual((batch_size, n_hidden), logits.shape) 
      losses = tf.get_collection('losses')
      self.assertEqual(len(losses), 1) 


if __name__ == '__main__': 
  tf.test.main()
      
