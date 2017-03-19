import tensorflow as tf

class MultiVocab(object): 
  ''' This class tracks vocabularies and embedding  variables 
      the vocab and feature file should have the first two lines be: 
      <PAD>
      <UNK>
      The default for unknown values will always be 1, and the PAD token
      will always be 0
  '''

  def __init__(self, vocab_path, feat_path, labels_path): 

    self.vocab = tf.contrib.lookup.string_to_index_table_from_file(
            vocabulary_file=vocab_path, num_oov_buckets=0, default_value=1)
    self.vocab_inv = tf.contrib.lookup.index_to_string_table_from_file(
            vocabulary_file=vocab_path, default_value="<UNK>")
    
    # total hack. tF makes the size a tensor, and cant use it to build graphs
    self.vocab_size = len(open(vocab_path).read().splitlines())

    self.feats = tf.contrib.lookup.string_to_index_table_from_file(
            vocabulary_file=feat_path, num_oov_buckets=0, default_value=1)
    self.feats_inv = tf.contrib.lookup.index_to_string_table_from_file(
            vocabulary_file=feat_path, default_value="<UNK>")
    self.feats_size = len(open(feat_path).read().splitlines())
  
    self.labels= tf.contrib.lookup.string_to_index_table_from_file(
            vocabulary_file=labels_path, num_oov_buckets=0, default_value=1)
    self.labels_inv= tf.contrib.lookup.index_to_string_table_from_file(
            vocabulary_file=labels_path, default_value="<UNK>")

    self.num_classes = len(open(labels_path).read().splitlines())
