import tensorflow as tf
import numpy as np
from nlpfromscratch.vocab import  MultiVocab
from nlpfromscratch.csvreader import CSVReader
import os 
curr_path = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(curr_path, 'sample')
class VocabReader(tf.test.TestCase): 
  

  def test_reader(self): 
    
    batch_size = 10
    word_dim = 3
    feat_dim = 2
    num_feats = 1
    dataset = os.path.join(data_dir, 'sample.csv')
    vocab_path = os.path.join(data_dir, 'vocab')
    feats_path = os.path.join(data_dir, 'feats')
    labels_path = os.path.join(data_dir, 'labels')

    multi_vocab = MultiVocab( vocab_path, feats_path, labels_path)
    csvreader = CSVReader(dataset, batch_size, num_feats)
    
    seq_len = csvreader.seq_len
    tokens_pl = tf.placeholder(tf.string, (batch_size, seq_len))
    enc_tokens = multi_vocab.vocab.lookup(tokens_pl)
    features_pl = tf.placeholder(tf.string, (batch_size, seq_len, num_feats ))
    enc_features = multi_vocab.vocab.lookup(features_pl)
    labels_pl = tf.placeholder(tf.string, (batch_size,) )
    enc_labels = multi_vocab.vocab.lookup(labels_pl)


    with self.test_session() as sess: 
      sess.run(tf.global_variables_initializer())
      sess.run(tf.tables_initializer())
      for tokens, features, labels in csvreader.batcher(1): 
        feed_dict = {tokens_pl:tokens, features_pl:features, labels_pl:labels}
        t,f,l= sess.run([enc_tokens, enc_features, enc_labels], feed_dict=feed_dict)
        self.assertAllEqual((batch_size,seq_len), t.shape)
        self.assertAllEqual((batch_size,seq_len, num_feats), f.shape)
        self.assertAllEqual((batch_size,), l.shape)
if __name__ == '__main__': 
  tf.test.main()
      
