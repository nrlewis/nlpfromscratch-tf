# nlpfromscratch-tf
Tensorflow implementation of NLP From Scratch (https://arxiv.org/abs/1103.0398) 

# Quickstart for Sentence Convolution
1. clone repo and get requirements (pandas mostly)
```bash
git clone https://github.com/nrlewis/nlpfromscratch-tf.git
cd nlpfromscratch-tf
pip install -r requirements.txt
```
2. prep CONLL data for POS Tagging. Creates 'caps' features anad sentences 
upto 40 tokens, split %30 into validation set
```bash
mkdir -p conlldata/sent # create logging directory
python prep_scripts/prep_conll.py data/conll2000/train.txt conlldata/sent -s 40 -v .3
```
3. run the training 
```bash
mkdir sent_conv_log
python train_from_scratch.py \
	sent_conv_log \
	conlldata/sent/train_pos_sent.csv \
	conlldata/sent/valid_pos_sent.csv \
	conlldata/sent/vocab.json
```
4. Check out tensorboard
```bash
cd sent_conv_log
tensorboard --logdir . 
```
		


