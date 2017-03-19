import tensorflow as tf

def class MultiVocab(object): 
  ''' This class tracks vocabularies and embedding  variables 
      the vocab and feature file should have the first two lines be: 
      <PAD>
      <UNK>

      The default for unknown values will always be 1, and the PAD token
      will always be 0
  '''

  def __init__(self, vocab_path, feat_path, label_path): 

    self.vocab = tf.contrib.lookup.string_to_index_table_from_file(
            vocabulary_file=vocab_path, num_oov_buckets=0, default_value=1)

    self.vocab_inv = tf.contrib.lookup.index_to_string_table_from_file(
            vocabulary_file=vocab_path, default_value="<UNK>")

    self.feats = tf.contrib.lookup.string_to_index_table_from_file(
            vocabulary_file=feat_path, num_oov_buckets=0, default_value=1)

    self.feat_inv = tf.contrib.lookup.index_to_string_table_from_file(
            vocabulary_file=feat_path, default_value="<UNK>")

    self.labels= tf.contrib.lookup.string_to_index_table_from_file(
            vocabulary_file=labels_path, num_oov_buckets=0, default_value=1)

    self.labels_inv= tf.contrib.lookup.index_to_string_table_from_file(
            vocabulary_file=labels_path, default_value="<UNK>")
