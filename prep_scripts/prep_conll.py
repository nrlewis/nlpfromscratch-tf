import sys
import argparse
import pandas as pd
import random
import string
import json 
PUNCT = set(string.punctuation)

PAD_TOKEN = '<PAD>'
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
  ''' pad and window sentence ''' 

  # add pad to top and bottom of sent
  
  pad = [ [PAD_TOKEN] * len(df.columns) ] *  winlen
  top_df = pd.DataFrame(data=pad, columns=df.columns)
  bottom_df = pd.DataFrame(data=pad, columns=df.columns)

  padded = pd.concat([top_df, df, bottom_df])
  padded.reset_index(drop=True, inplace=True)
  
  word_cols= ['w_%d' %w for w in range(-winlen, winlen+1)]
  feature_cols = ['cap_%d' %w for w in range(-winlen, winlen+1)]
  if extra_feat: 
    extra_feats  = ['%s_%d' %(extra_feat, w) for w in range(-winlen, winlen+1)]
    feature_cols.extend(extra_feats)
  
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

def prep_sentence_conv(df): 
  ''' 
  output sentence for convolution.  each sentence is output with 
  relative position information from the classifying token
  
  output for 'The quick brown fox' for a POS tagger. the label at the 
  end applies to the token at pos_0

  The quick brown fox TITLE pos_0  LOWER pos_1  LOWER pos_2  LOWER pos_3 DET 
  The quick brown fox TITLE pos_-1 LOWER pos_0  LOWER pos_1  LOWER pos_2 ADJ 
  The quick brown fox TITLE pos_-2 LOWER pos_-1 LOWER pos_0  LOWER pos_1 ADJ 
  The quick brown fox TITLE pos_-3 LOWER pos_-2 LOWER pos_-1 LOWER pos_0 NOUN 

  Also put padding for  sents below MAX_SEQ_LEN and truncate if over

  '''
  # Median sentence len in training set is 24 tokens, 90 percentile is 40, 
  SEQ_LEN = 40

  sent_ids = df.sent_id.unique()
  dfs = []

  for sent_id in sent_ids: 
    print 'expanding sentence %d' %sent_id
    sent_df = df[df.sent_id == sent_id]
    windowed = sent_expand(sent_df, SEQ_LEN)
    dfs.append(windowed)

  return dfs



def split_and_print(df, suffix, args):
  random.shuffle(dfs)
  train_size = int(len(dfs) * (1 - args.valid))
  train = dfs[:train_size]
  valid = dfs[train_size:]
  train_df = pd.concat(train)
  train_df.to_csv('train' + suffix, index=False)
  valid_df = pd.concat(valid)
  valid_df.to_csv('valid' + suffix, index=False)



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

  if args.vocabjson: 
    vocab = ['<PAD>','<UNK>'] +  sorted(df['token'].unique())
    labels = sorted(df['label'].unique())
    features = ['<PAD>','<UNK>'] +  sorted(df['caps'].unique())
    outjson = {'vocab':vocab, 'features':features, 'labels':labels}
    with open('vocab.json', 'w') as out: 
       out.write(json.dumps(outjson, indent=2))
  
  dfs = None
  suffix = ''
  if args.sentence: 
    dfs = prep_sentence_conv(df)
    suffix = '_sent.csv'
  else:
    dfs = prep_window(df)
    suffix = '_w%d.csv'%args.winlen

  if args.valid: 
    split_and_print(dfs, suffix)
  else: 
    all_dfs = pd.concat(dfs)
    all_dfs.to_csv(args.inpath + suffix, index=False)
    

if __name__ == '__main__': 
  ''' convert data set from conll format and split into train / valid ''' 
  parser = argparse.ArgumentParser()
  parser.add_argument('inpath', type=str)
  parser.add_argument('-l', '--label', choices=['pos', 'chunk'], default='pos')

  parser.add_argument('-e', '--extra_feat', choices=['pos', 'chunk'], default='pos', 
      help='add an extra feature to the output (useful to add pos to chunk) ')
  parser.add_argument('-w', '--winlen', type=int, default=3, 
      help='length of the window')
  parser.add_argument('-v', '--valid', type=float, default=None, 
      help='proportion to split into valid set')
  parser.add_argument('-j', '--vocabjson', action='store_true', default=True, 
      help='dump a file for the vocabulary vocab.json')

  parser.add_argument('-s', '--sentence', type=int, default=0, 
      help='Prepare conll data for sentence convolution')

  args = parser.parse_args()
  main(args)
  


