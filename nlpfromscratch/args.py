import argparse

parser = argparse.ArgumentParser()
parser.add_argument('log_dir', type=str, help='log directory for storage')
parser.add_argument('train_path', type=str, help='path to train file')
parser.add_argument('valid_path', type=str, help='path to valid file')
parser.add_argument('vocab_path', type=str, help='path to vocab json')

parser.add_argument('-r', '--lambda_', type=float, default=0.001, 
    help='Regularization weight')
parser.add_argument('-l', '--init_lr', type=float, default=0.01, 
    help='Initial Learning Rate')
parser.add_argument('-w', '--word_dim', type=int, default=50, 
    help='dimension of word embeddings')
parser.add_argument('-f', '--feat_dim', type=int, default=5, 
    help='dimension of feature  embeddings')
parser.add_argument('-1', '--n_hidden', type=int, default=300, 
    help='number of nodes in hiddent layer')
parser.add_argument('-b', '--batch_size', type=int, default=32, 
    help='Batch Size')
parser.add_argument('-m', '--max_epochs', type=int, default=20, 
    help='Max number of epochs')
