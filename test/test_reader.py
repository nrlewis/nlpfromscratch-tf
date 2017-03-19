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
    dataset = os.path.join(data_dir, 'train_w3_sample.csv')
    vocab_path = os.path.join(data_dir, 'sample_vocab.json')

    multi_vocab = MultiVocab( vocab_path)
    csvreader = CSVReader(dataset, batch_size)
    
    seq_len = csvreader.seq_len
    num_feats = csvreader.num_feats
    
    tokens_pl = tf.placeholder(tf.string, (batch_size, seq_len))
    enc_tokens = multi_vocab.vocab.lookup(tokens_pl)

    features_pl = tf.placeholder(tf.string, (batch_size, seq_len, num_feats ))
    enc_features = multi_vocab.feats.lookup(features_pl)

    labels_pl = tf.placeholder(tf.string, (batch_size,) )
    enc_labels = multi_vocab.labels.lookup(labels_pl)

    # use later
    dec_tokens = multi_vocab.vocab_inv.lookup(enc_tokens)
    dec_features = multi_vocab.feats_inv.lookup(enc_features)
    dec_labels = multi_vocab.labels_inv.lookup(enc_labels)


    with self.test_session() as sess: 
      sess.run(tf.global_variables_initializer())
      sess.run(tf.tables_initializer())
      for tokens, features, labels in csvreader.batcher(1): 
        #print tokens[0]
        #print features[0]
        feed_dict = {tokens_pl:tokens, features_pl:features, labels_pl:labels}
        t,f,l= sess.run([enc_tokens, enc_features, enc_labels], feed_dict=feed_dict)
        #print t[0]
        #print f[0]
        self.assertAllEqual((batch_size,seq_len), t.shape)
        self.assertAllEqual((batch_size,seq_len, num_feats), f.shape)
        self.assertAllEqual((batch_size,), l.shape)
if __name__ == '__main__': 
  tf.test.main()
      
