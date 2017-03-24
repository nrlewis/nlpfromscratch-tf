import tensorflow as tf
import json 

class MultiVocab(object): 
  ''' This class tracks vocabularies and embedding  variables 
      the vocab and feature file should have the first two lines be: 
      <PAD>
      <UNK>
      The default for unknown values will always be 1, and the PAD token
      will always be 0
    
    For this class I went with the TF implementations, but I'm not sure
    it's usefule if I don't combine it with TFRecords
  '''

  def __init__(self, vocab_path):
    vocab_j = json.loads(open(vocab_path).read())
    vocab_t = tf.constant(vocab_j['vocab'])
    self.vocab = tf.contrib.lookup.string_to_index_table_from_tensor(
            mapping=vocab_t, num_oov_buckets=0, default_value=1)
    self.vocab_inv = tf.contrib.lookup.index_to_string_table_from_tensor(
            mapping=vocab_t, default_value="<UNK>")
    
    # total hack. tF makes the size a tensor, and cant use it to build graphs
    self.vocab_size = len(vocab_j['vocab'])

    feats_t = tf.constant(vocab_j['features'])
    self.feats = tf.contrib.lookup.string_to_index_table_from_tensor(
            mapping=feats_t, num_oov_buckets=0, default_value=1)
    self.feats_inv = tf.contrib.lookup.index_to_string_table_from_tensor(
            mapping=feats_t, default_value="<UNK>")
    self.feats_size = len(vocab_j['features'])

    self.labels_t = tf.constant(vocab_j['labels']) 
    self.labels= tf.contrib.lookup.string_to_index_table_from_tensor(
            mapping=self.labels_t, num_oov_buckets=0, default_value=1)
    self.labels_inv= tf.contrib.lookup.index_to_string_table_from_tensor(
            mapping=self.labels_t, default_value="<UNK>")

    self.num_classes = len(vocab_j['labels'])
