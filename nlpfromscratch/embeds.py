import tensorflow as tf

class Embeddings(object): 
  
  def __init__(self, multi_vocab, word_dim, feat_dim, num_feats): 
    
    self.multi_vocab = multi_vocab
    self.word_dim = word_dim
    self.feat_dim = feat_dim
    self.num_feats = num_feats
    self._init_embeddings()

  def _init_embeddings(self): 
    
    with tf.variable_scope('Embeddings'):
      self.word_embs = tf.get_variable('word_embs', 
          (self.multi_vocab.vocab_size, self.word_dim),
          initializer = tf.contrib.layers.xavier_initializer()
          )

      self.feat_embs = tf.get_variable('feat_embs', 
          (self.multi_vocab.feats_size, self.feat_dim),
          initializer = tf.contrib.layers.xavier_initializer()
          )

  def encode(self, tokens, features):   

    shape = tokens.get_shape()
    batch_size = shape[0].value
    num_toks = shape[1].value 
    
    with tf.name_scope('WindowedEncoder'):

      # convert strings to ids
      encoded_tokens = self.multi_vocab.vocab.lookup(tokens)
      encoded_features = self.multi_vocab.feats.lookup(features)

      # reshap features so that we have a list of features per token
      shaped_features = tf.reshape(encoded_features, (batch_size, num_toks, self.num_feats))

      # Convert int token ids to dense vectors via lookup table
      word_lookup = tf.nn.embedding_lookup(self.word_embs, encoded_tokens)
      feat_lookup = tf.nn.embedding_lookup(self.feat_embs, encoded_features)

      # Concatnate dense vectors into single input vector (batched)
      # first reshape feature vectors, concating along last dimension
      feat_flat = tf.reshape(feat_lookup, (batch_size, num_toks, -1))
      word_feats = tf.concat([word_lookup, feat_flat], axis=2)
      input_seq = tf.reshape(word_feats, [batch_size, -1])


    return input_seq
