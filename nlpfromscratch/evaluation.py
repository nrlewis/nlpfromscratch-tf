from collections import defaultdict
import tensorflow as tf

def metrics(labels): 
  ''' 
  create variables for each class label.  This way we can track 
  precision / recall / f1 scores during training
  '''

  with tf.variable_scope('metrics'): 

    for metric in ['precision', 'recall', 'f1']:
      with tf.variable_scope(metric): 
        for label in labels: 
            m = tf.get_variable(label, [], initializer=tf.constant_initializer(0.0))
            tf.summary.scalar(m.name, m)
            tf.add_to_collection('metrics', m)          

        m = tf.get_variable('average', [], initializer=tf.constant_initializer(0.0))
        tf.add_to_collection('metrics', m)          
        tf.summary.scalar(m.op.name, m)
      
def prf_eval(valid_reader, predict, multi_vocab, sess, tokens_pl, features_pl):
  ''' 
  perform an evaluation for each input.  Calculation precision, recall, and f1
  '''

  tp_map = defaultdict(lambda : 0)
  fp_map = defaultdict(lambda : 0)
  fn_map = defaultdict(lambda : 0)

  
  for tokens, features, truths, in valid_reader.batcher(1): # only one epoch for validation
    feed_dict = {tokens_pl:tokens, features_pl:features}
    predictions = sess.run(predict, feed_dict=feed_dict)

    # sanity check
    len(predictions) == len(truths)

    for prediction, truth  in zip(predictions, truths):
      if truth == prediction:
        tp_map[truth] += 1
      else: 
        fp_map[prediction] +=1
        fn_map[truth] += 1

  return set_scores(tp_map, fp_map, fn_map, multi_vocab._labels, sess)
  
def set_scores(tp_map, fp_map, fn_map, labels, sess):
  
  # calculate precision, recall and f1
  p_avg = 0.0 
  r_avg = 0.0 
  f1_avg = 0.0

  with tf.variable_scope('metrics', reuse=True):

    count = 0.0
    for label in labels: 
      tp = tp_map[label]
      fp = fp_map[label]
      fn = fn_map[label]
    
      p_var = tf.get_variable('precision/%s' %label)
      r_var = tf.get_variable('recall/%s' %label)
      f1_var  = tf.get_variable('f1/%s' %label)
      
      p = tp / float(tp + fp) if tp > 0 or fp > 0 else 0.0
      r = tp / float(tp + fn) if tp > 0 or fp > 0 else 0.0
      f1 =  2 * ( (p * r)  / (p + r) ) if p > 0 or r > 0 else 0.0

      sess.run([p_var.assign(p), r_var.assign(r), f1_var.assign(f1)])


      if label != 'O': 
        p_avg += p
        r_avg += r
        f1_avg += f1
        count += 1.0

   
    p_avg /= count
    r_avg /= count
    f1_avg /= count
     
    p_avg_var = tf.get_variable('precision/average')   
    r_avg_var = tf.get_variable('recall/average')   
    f1_avg_var = tf.get_variable('f1/average')   

    sess.run([p_avg_var.assign(p_avg), r_avg_var.assign(r_avg),
      f1_avg_var.assign(f1_avg)])

    return p_avg, r_avg, f1_avg
