from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np
from nlpfromscratch.vocab import  MultiVocab
from nlpfromscratch.evaluation import metrics, set_scores
import os 

curr_path = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(curr_path, 'sample')


class TestEvaluation(tf.test.TestCase):


  def test_init_and_set_scores(self): 
    
    tp_map = { 'A':3, 'B':2, 'C':4, 'O':10}
    fp_map = { 'A':1, 'B':6, 'C':0, 'O':2}
    fn_map = { 'A':4, 'B':1, 'C':0, 'O':4} 

    labels = tp_map.keys()

    metrics(labels)
  
  
    expected_num_vars = ( len(labels) + 1 ) * 3 
    with self.test_session() as sess: 
      sess.run(tf.global_variables_initializer())
      metric_vars = tf.get_collection('metrics')
      self.assertAllEqual(expected_num_vars, len(metric_vars))
      
      for m in metric_vars: 
        self.assertEqual(0.0, m.eval())

      set_scores(tp_map, fp_map, fn_map, labels, sess)
     
      metric_vars = tf.get_collection('metrics')
      
      for m in metric_vars: 
        self.assertNotEqual(0.0, m.eval())

if __name__ == '__main__': 
  tf.test.main()
      
