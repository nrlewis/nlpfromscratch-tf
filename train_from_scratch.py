import tensorflow as tf
import numpy as np
from nlpfromscratch.vocab import  MultiVocab
from nlpfromscratch.csvreader import CSVReader
from nlpfromscratch.embeds import Embeddings
from nlpfromscratch.mlp import linear
from nlpfromscratch.loss import reg_softmax_loss
from nlpfromscratch.args import parser
from nlpfromscratch.convnet import conv_max
from nlpfromscrtach.evaluation import metrics, prf_eval
import os 


def run(FLAGS):     

  # helper classes
  train_reader = CSVReader(FLAGS.train_path, FLAGS.batch_size)
  valid_reader = CSVReader(FLAGS.valid_path, FLAGS.batch_size)
  seq_len = train_reader.seq_len
  num_feats = train_reader.num_feats

  with tf.Graph().as_default():


    sess = tf.Session() 
    global_step = tf.contrib.framework.get_or_create_global_step()

    # placeholders  
    tokens_pl = tf.placeholder(tf.string, (FLAGS.batch_size, seq_len))
    features_pl = tf.placeholder(tf.string, (FLAGS.batch_size, seq_len, num_feats ))
    labels_pl = tf.placeholder(tf.string, (FLAGS.batch_size,) )

    # vocabulary lookups
    multi_vocab = MultiVocab(FLAGS.vocab_path)
    label_lookup = multi_vocab.labels.lookup(labels_pl)
    metrics(multi_vocab.labels_inv.keys())    
    # init word emebedings
    embeddings = Embeddings(multi_vocab, FLAGS.word_dim, FLAGS.feat_dim, num_feats)
    encoded_input = embeddings.encode(tokens_pl, features_pl)

    if FLAGS.sent_conv:
      # for convlution, reshape input to fit conv2d function and pass
      conv_encoded = tf.reshape(encoded_input, (FLAGS.batch_size, seq_len, -1, 1))
      sent_encoding = conv_max(conv_encoded, FLAGS.kernel_ht, FLAGS.n_kernels)
      
      # this is ugly to rename a computational graph operation, but .. prettire
      # than a new script 
      encoded_input = sent_encoding

   
    # MLP Feed Forward
    wx_plus_b = linear(encoded_input, FLAGS.n_hidden, 'hidden', FLAGS.lambda_)
    hidden = tf.nn.relu(wx_plus_b)
    logits = linear(hidden, multi_vocab.num_classes, 'classify', FLAGS.lambda_)
    predict = multi_vocab.labels_inv.lookup(tf.nn.softmax(logits))

    # Loss and Train
    loss = reg_softmax_loss(logits, label_lookup)
    grad = tf.train.GradientDescentOptimizer(FLAGS.init_lr)
    train_op = grad.minimize(loss, global_step=global_step)   

    sess.run(tf.global_variables_initializer())
    sess.run(tf.tables_initializer())
    for tokens, features, labels in train_reader.batcher(FLAGS.max_epochs): 
      feed_dict = {tokens_pl:tokens, features_pl:features, labels_pl:labels}
      step_loss, gs, _   = sess.run([loss, global_step, train_op], 
                                    feed_dict=feed_dict)
      
      if gs % 100 == 0: 
        print gs, step_loss

if __name__ == '__main__': 
  
    FLAGS=parser.parse_args() 
    run(FLAGS) 
      
