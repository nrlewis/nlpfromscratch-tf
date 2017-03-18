import tensorflow as tf
import numpy as np
from nlpfromscratch.loss import add_loss_var, softmax_loss

class LossTest(tf.test.TestCase): 
  
  def test_zero_loss(self): 
    
    lambda_ = 0.001
    zero = tf.Variable(np.zeros((10,10), dtype=np.float32))
    add_loss_var(zero, lambda_ )
    expected_loss = 0.0 
    with self.test_session() as sess: 
      sess.run(tf.global_variables_initializer())
      losses = tf.get_collection('losses') 
      loss = losses[0].eval()
      self.assertEqual(expected_loss,  loss)
      
  def test_non_zero_loss(self): 
    
    lambda_ = 0.001
    n = 10
    init = np.zeros((n,n), dtype=np.float32) + 2
    twos = tf.Variable(init)
    add_loss_var(twos, lambda_ )
     
    expected_loss = (np.sum(init**2) / 2.) * lambda_

    with self.test_session() as sess: 
      sess.run(tf.global_variables_initializer())
      losses = tf.get_collection('losses') 
      loss = losses[0].eval()
      self.assertAlmostEqual(expected_loss, loss)

  def test_negative(self): 
    
    lambda_ = 0.001
    n = 10
    twos_init = np.zeros((n,n), dtype=np.float32) - 2 
    threes_init = np.zeros((n,n), dtype=np.float32) - 3
    twos= tf.Variable(twos_init)
    threes= tf.Variable(threes_init)
    add_loss_var(twos, lambda_ )
    add_loss_var(threes, lambda_ )
     
    expected_loss = (np.sum(twos_init**2) / 2.)  * lambda_
    expected_loss += (np.sum(threes_init**2) / 2.)  * lambda_

    with self.test_session() as sess: 
      sess.run(tf.global_variables_initializer())
      losses = tf.get_collection('losses') 
      self.assertEqual(2, len(losses))
      total_loss = sum(l.eval() for l in losses)
      self.assertAlmostEqual(expected_loss, np.sum(total_loss))

  def test_multiple_losses(self): 
    
    lambda_ = 0.001
    n = 10
    twos_init = np.zeros((n,n), dtype=np.float32) + 2
    threes_init = np.zeros((n,n), dtype=np.float32) + 3
    twos= tf.Variable(twos_init)
    threes= tf.Variable(threes_init)
    add_loss_var(twos, lambda_ )
    add_loss_var(threes, lambda_ )
     
    expected_loss = (np.sum(twos_init**2) / 2.)  * lambda_
    expected_loss += (np.sum(threes_init**2) / 2.)  * lambda_

    with self.test_session() as sess: 
      sess.run(tf.global_variables_initializer())
      losses = tf.get_collection('losses') 
      self.assertEqual(2, len(losses))
      total_loss = sum(l.eval() for l in losses)
      self.assertAlmostEqual(expected_loss, np.sum(total_loss))

  def test_softmax_loss(self): 
    batch_size = 10
    nclasses = 3
    lambda_ = 0.001
    emb_dim = 10
    
    logits = np.random.uniform(-10, 10, size=(batch_size, nclasses)).astype(np.float32)
    labels = np.random.randint(0,2, batch_size)

    twos = tf.Variable(np.zeros((emb_dim, emb_dim), dtype=np.float32) + 2)
    add_loss_var(twos, lambda_ )

    with self.test_session() as sess:

      sess.run(tf.global_variables_initializer())
      losses = tf.get_collection('losses') 
      total_loss = sess.run(softmax_loss(logits, labels))
      self.assertEqual( np.float32, type(total_loss))
      self.assertEqual( (), total_loss.shape)

    

if __name__ == '__main__': 
  tf.test.main()
      
