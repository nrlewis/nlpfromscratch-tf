from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import random 
import pandas as pd
import numpy as np

class CSVReader(object):

  def __init__(self, filepath, batch_size):
    ''' Read in CSV. 
    The format of the csv fshould be: 
    <tokens><features><label>.  ie., 

    if there are N words, and I features per word :
    w1,w2,w3,...,wn,f1_1,f1_2,f2_1,f2_2,...fn_i,label

    i.e, lets say we had two features per word, capitialization and suffix, 
        and a part of speech label for the center word: 
    
    w_-1, w_0, w_1, f_caps_-1, f_suff_-1, f_caps_0, f_suff_0, f_caps_1, f_suff_1, label
    the , dog, ate, TITLE  , he     , LOWER , og    , LOWER , at    , NOUN
    
    w_0 is the word to tag with the label. caps_0 and suff_0 are its
    capitalization and suffix, respectively
   
    The headers for words MUST be prefix w_<i>, a "w" followed relative position
    of the center. 

    the headers for features MUST be prefix <feat_name>_<i>.
    '''
    
    self.filepath = filepath
    self.batch_size = batch_size  
    self.df = pd.read_csv(filepath)
    
    self.word_cols = [c for c in self.df.columns if c.startswith('w_')]
    self.seq_len = len(self.word_cols)
    non_feats = set(self.word_cols + ['label'])
    self.feat_cols = [c for c in self.df.columns if c.startswith('f_')]
    self.num_feats = int(len(self.feat_cols) / len(self.word_cols))

    # sanity check
    assert len(self.feat_cols) == len(self.word_cols) * self.num_feats
     
    
  def batcher(self, num_epochs): 
    self.epoch = 0  
    df = self.df
    while self.epoch < num_epochs: 
      indices = range(len(df))
      random.shuffle(indices)
      
      num_batches = len(df) // self.batch_size
      last_row = num_batches * self.batch_size  

      for i in range(0, last_row, self.batch_size):
        #print i, i+self.batch_size
        batch = df.loc[indices[i:i+self.batch_size]]
        tokens = batch[self.word_cols].as_matrix()
        features = batch[self.feat_cols].as_matrix()
        features = np.reshape(features, (self.batch_size, self.seq_len, self.num_feats))
        
        
        labels = batch['label'].values
        yield tokens, features, labels
     

      self.epoch += 1 

if __name__ == '__main__': 
  import sys
  csvreader = CSVReader(sys.argv[1], 9)
  feats = []
  toks = []
  labels = []


  for t,f,l in csvreader.batcher(1): 
    print( t)
    print( f)
    print( l)
