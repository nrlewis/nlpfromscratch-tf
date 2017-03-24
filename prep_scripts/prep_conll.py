import os
import numpy as np
import sys
import argparse
import pandas as pd
import random
import string
import json 
PUNCT = set(string.punctuation)

PAD_TOKEN = '<PAD>'
seq_len = 40 

def almost(row):
  ''' apply the "almost" part of almost from scratch.  create a column
  that contains the tokens capitalization .  Only difference from acutal 
  NLP (almost) from scratch is that I'm adding punctuation and digit 
  information'''

  if row.token.islower(): 
    return 'LOWER'
  elif row.token.isupper(): 
    return 'UPPER'
  elif row.token.istitle(): 
    return 'TITLE'
  elif row.token.isdigit():
    return 'DIGIT'
  elif row.token in PUNCT: 
    return 'PUNCT'
  else:
    return 'MIXED'

def window_sent(df, winlen, extra_feat=None): 
  ''' pad and window sentence 
    
  Example for winlen of 1 for Part of Speech Tagging

  In: The quick fox
  Out: 
  <PAD> the quick <PAD> TITLE LOWER  DET
  the quick fox  TITLE LOWER LOWER ADJ
  quick fox <PAD> LOWER LOWER LOWER NOUN 

  ''' 

  # add pad to top and bottom of sent
  
  pad = [ [PAD_TOKEN] * len(df.columns) ] *  winlen
  top_df = pd.DataFrame(data=pad, columns=df.columns)
  bottom_df = pd.DataFrame(data=pad, columns=df.columns)

  padded = pd.concat([top_df, df, bottom_df])
  padded.reset_index(drop=True, inplace=True)
  
  word_cols= ['w_%d' %w for w in range(-winlen, winlen+1)]
  feature_cols = ['f_cap_%d' %w for w in range(-winlen, winlen+1)]

  new_columns = word_cols + feature_cols + ['label']
  #print new_columns 
  data = []
  for i in range(winlen, len(padded) - winlen): 
    win_df = padded.loc[i-winlen:i+winlen]
    tokens = win_df.token.values
    caps = win_df.caps.values
    if extra_feat: 
      feats = win_df[extra_feat].values
      caps.extend(feats)
    
    label = win_df.loc[i, 'label']
    row = []
    row.extend(tokens)
    row.extend(caps)
    row.append(label)
    data.append(row)

  newdf = pd.DataFrame(data=data, columns=new_columns)
  return newdf

def prep_window(df): 

  sent_ids = df.sent_id.unique()
  dfs = []

  for sent_id in sent_ids: 
    print 'windowing sentence %d' %sent_id
    sent_df = df[df.sent_id == sent_id]
    windowed = window_sent(sent_df, args.winlen)
    dfs.append(windowed)

  return dfs

def expand_sent(df, seq_len, position_feats): 
  ''' 
  output sentence for convolution.  each sentence is output with 
  relative position information from the classifying token
  
  
  Here is an example for a sentence of sent_len of 3, but the CNN input 
  seq_len is 4, so we place a pad at the end of the sentence: 
  
  Input Sentence: 'The quick fox' # sent_len == 3
  Padded Sentence: 'The quick fox <PAD>' # seq_len  == 4

  Below is the padded sentence input for a POS tagger. The format is 
  <words> <features> <label>, where each ith feature applies to the ith word. 
  
  w_0,w_1,...,w_n,f_00,f_01,f_10,f_11,...,f_ij,label

   
  The label at the end applies to the token at pos_0.  Because we don't want 
  to label the  <PAD>, we stop the position vectors before then. 

  The quick fox <PAD> TITLE pos_0  LOWER pos_1  LOWER pos_2  LOWER pos_3 DET 
  The quick fox <PAD> TITLE pos_-1 LOWER pos_0  LOWER pos_1  LOWER pos_2 ADJ 
  The quick fox <PAD> TITLE pos_-2 LOWER pos_-1 LOWER pos_0  LOWER pos_1 ADJ 
  # note there is no 4th line here #
  
  If sent_len > seq_len, then truncate the sentence instead of padding.
   
  Out Shape: ( min(sent_len, seq_len), seq_len + (num_features * seq_len) + 1) 

  '''
  # pad sentence
  sent_len = len(df)
  sent = df.token.values
  caps = df.caps.values
  labels = df.label.values

  if sent_len < seq_len:
    
    pad_len = seq_len - sent_len
    pad = np.array([PAD_TOKEN] * pad_len)
    sent = np.concatenate([sent,pad], -1)
    caps = np.concatenate([caps,pad], -1)

  elif sent_len > seq_len:
    sent = sent[:seq_len]
    caps = caps[:seq_len]  
    labels = labels[:seq_len]
     
  # determine the number of examples per sentence
  num_examples = min(seq_len, sent_len)
  token_matrix = np.reshape(np.tile(sent,num_examples), (num_examples, -1))

  # merge the two features first
  caps_matrix = np.reshape(np.tile(caps,num_examples), (num_examples, seq_len, 1))
  positions = np.reshape(position_feats[:num_examples,:], (num_examples, seq_len, 1))
  features = np.reshape(np.concatenate([caps_matrix, positions], 2), (num_examples, -1))
 
  # concat tokens and features 
  data = np.concatenate([token_matrix, features], 1)

  # column headers
  words = ['w_%d' %i for i in range(seq_len)] # words
  caps =['f_caps_%d' %i for i in range(seq_len)]  #caps
  pos =['f_pos_%d' %i for i in range(seq_len)]  # pos
  feat_cols = [c for pair in zip(caps, pos) for c in pair]
  cols = words + feat_cols 
  
  new_df = pd.DataFrame(data=data, columns =cols) 
  new_df['label'] = labels 

  return new_df
  
  
   
  

def prep_sentence_conv(df, seq_len): 
  # Median sentence len in training set is 24 tokens, 90 percentile is 40, 

  sent_ids = df.sent_id.unique()
  dfs = []

  # Generate the matrix of positions  [ [0,1,2],[-1,0,1], [-2,-1,0]]
  positions = np.reshape(np.tile(np.arange(seq_len),seq_len), (seq_len, -1))
  positions -= positions.T
  # change to strings for output [ ['pos_0', 'pos_1', 'pos_2'],..]
  to_str_feat = np.vectorize(lambda x:'pos_%d'%x)
  position_feats = to_str_feat(positions)
  
  for sent_id in sent_ids: 
    print 'expanding sentence %d' %sent_id
    sent_df = df[df.sent_id == sent_id]
    expanded = expand_sent(sent_df,seq_len, position_feats)
    dfs.append(expanded)

  return dfs



def split_and_print(dfs, suffix, args):

  random.shuffle(dfs)
  train_size = int(len(dfs) * (1 - args.valid))
  train = dfs[:train_size]
  valid = dfs[train_size:]
  train_df = pd.concat(train)
  train_path = os.path.join(args.outdir, 'train'+suffix)
  train_df.to_csv(train_path, index=False)
  valid_df = pd.concat(valid)
  valid_path = os.path.join(args.outdir, 'valid'+suffix)
  valid_df.to_csv(valid_path, index=False)



def main(args):
  df = pd.read_csv(args.inpath, sep=' ', skip_blank_lines=False, names=['token', 'pos', 'chunk'])

  # split into sentence from blank rows
  df['sent_id'] = df.T.isnull().all().cumsum()
  df.dropna(axis=0, inplace=True)
  # extract "almost" features
  df['caps'] = df.apply(lambda row: almost(row),axis=1)
  # set all tokens to lower case
  df['token'] = df.token.str.lower()
  
  # set label and remove unecessary cols
  df['label'] = df[args.label]
  # window the input by sentence

  # print out the features for use in TF
  if args.vocabjson: 
    vocab = ['<PAD>','<UNK>'] +  sorted(df['token'].unique())
    labels = sorted(df['label'].unique())
    features = ['<PAD>','<UNK>'] +  sorted(df['caps'].unique())
    if args.sentence: 
      seq_len = args.sentence
      features += ['pos_%d'%i for i in range(-seq_len, seq_len+1)]

    outjson = {'vocab':vocab, 'features':features, 'labels':labels}
    outpath = os.path.join(args.outdir, 'vocab.json')
    with open(outpath, 'w') as out: 
       out.write(json.dumps(outjson, indent=2))
  
  dfs = None
  suffix = ''

  if args.sentence: 
    seq_len = args.sentence
    dfs = prep_sentence_conv(df, seq_len)
    suffix = '_sent.csv'
  else:
    dfs = prep_window(df)
    suffix = '_w%d.csv'%args.winlen

  if args.valid: 
    split_and_print(dfs, suffix, args)
  else: 
    all_dfs = pd.concat(dfs)
    outpath = os.path.join(args.outdir,os.path.basename(args.inpath))
    all_dfs.to_csv( outpath+ suffix, index=False)
    

if __name__ == '__main__': 
  ''' convert data set from conll format and split into train / valid ''' 
  parser = argparse.ArgumentParser()
  parser.add_argument('inpath', type=str)
  parser.add_argument('outdir', type=str)
  parser.add_argument('-l', '--label', choices=['pos', 'chunk'], default='pos')

  parser.add_argument('-e', '--extra_feat', choices=['pos', 'chunk'], default='pos', 
      help='NOT IMPLEMENTED YET add an extra feature to the output (useful to add pos to chunk) ')
  parser.add_argument('-w', '--winlen', type=int, default=3, 
      help='length of the window')
  parser.add_argument('-v', '--valid', type=float, default=None, 
      help='proportion to split into valid set')
  parser.add_argument('-j', '--vocabjson', action='store_true', default=True, 
      help='dump a file for the vocabulary vocab.json')

  parser.add_argument('-s', '--sentence', type=int, default=None, 
      help='Prepare conll data for sentence convolution')

  args = parser.parse_args()
  main(args)
  


