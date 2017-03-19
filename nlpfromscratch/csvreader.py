import random 
import pandas as pd
import numpy as np

class CSVReader(object):

  def __init__(self, filepath, batch_size, num_feats):
    self.filepath = filepath
    self.batch_size = batch_size  
    self.num_feats = num_feats
    self.df = pd.read_csv(filepath)
    
    self.word_cols = [c for c in self.df.columns if c.startswith('w_')]
    self.seq_len = len(self.word_cols)
    self.feat_cols = [c for c in self.df.columns if not c.startswith('w_') and not c == 'label']  
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
    print t
    print f
    print l
