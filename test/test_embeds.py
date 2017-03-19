import tensorflow as tf
import numpy as np
from nlpfromscratch.vocab import  MultiVocab
from nlpfromscratch.csvreader import CSVReader
from nlpfromscratch.embeds import Embeddings
import os 
curr_path = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(curr_path, 'sample')


class ReaderAndEmbedding(tf.test.TestCase): 
  

  def test_windowing(self): 
    
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
    embeddings = Embeddings(multi_vocab,word_dim, feat_dim, num_feats)

    seq_len = csvreader.seq_len

    tokens_pl = tf.placeholder(tf.string, (batch_size, seq_len))
    features_pl = tf.placeholder(tf.string, (batch_size, seq_len, num_feats ))
    labels_pl = tf.placeholder(tf.string, (batch_size,) )
    
    encoded_input = embeddings.encode(tokens_pl, features_pl)

    input_dim = (seq_len * word_dim) + (seq_len * feat_dim * num_feats)
    with self.test_session() as sess: 
      sess.run(tf.global_variables_initializer())
      sess.run(tf.tables_initializer())
      for tokens, features, labels in csvreader.batcher(1): 
        feed_dict = {tokens_pl:tokens, features_pl:features, labels_pl:labels}
        input_ = sess.run([encoded_input], feed_dict=feed_dict)
        print input_    
        self.assertAllEqual((batch_size, input_dim), input_[0].shape)
        

if __name__ == '__main__': 
  tf.test.main()
      
