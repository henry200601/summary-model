import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
import pickle
from tensorflow.keras.models import load_model
from argparse import ArgumentParser 
parser=ArgumentParser()
parser.add_argument('--valid_data_path')
args=parser.parse_args()

test = pd.read_json(args.valid_data_path, lines=True)
test=test.filter(items=['id','text'])

with open("vocabularyattention.txt", "rb") as fp:
 vocabulary=pickle.load(fp)
with open("vocabularyattentionindex_word.txt", "rb") as fp:
 index_word=pickle.load(fp)

encoder_model=load_model('attbestencoder01.h5')
decoder_model=load_model('attbestdecoder01.h5')

text_tokenizer=Tokenizer(num_words=20000)
text_tokenizer.word_index = vocabulary.copy()
test['text']=text_tokenizer.texts_to_sequences(test['text'])
ytext=pad_sequences(test['text'],  maxlen=300, padding='post', truncating='post') 
test['predict']=''#把每句長度紀錄下來
res=np.zeros((len(test),40))
attentions=np.zeros((len(test),40,300))
for i in tqdm(range(100)):  
  e_out, e_h, e_c = encoder_model.predict(ytext[200*i:200*(i+1)],batch_size=256)
  target_seq = np.zeros((200,1))
  target_seq[:, 0] = text_tokenizer.word_index['bos']

  for j in range(40):
    output_tokens, h, c,attention_weight= decoder_model.predict([target_seq[:,-1]] + [e_out,e_h, e_c],batch_size=256)
    sampled_token_index = np.argmax(output_tokens,axis=-1)
    res[200*i:200*(i+1),j]=target_seq[:,0]
    attentions[200*i:200*(i+1),j,:]=attention_weight[:,0,:]
    target_seq = np.zeros((200,1))
    target_seq[:, 0] = sampled_token_index[:,0]
    e_h, e_c=h,c

query=1370
fig, ax = plt.subplots(figsize=(9,9))
ax.imshow(attentions[query][:25,:25])
x=[index_word[inp] if inp != 0 else '<pad>'  for inp in ytext[query][:25]]
y=[index_word[inp] if inp != 0 else '<pad>'   for inp in res[query,1:22] ]
for i in range(len(y)) :
  if y[i]=='eos':
     y[i]='<EOS>'
     
cax = ax.matshow(attentions[query][:25,:25])
fig.colorbar(cax)

ax.set_xticks(np.arange(len(x)))
ax.set_yticks(np.arange(len(y)))

ax.set_xticklabels(x)
ax.set_yticklabels(y)
ax.set_yticks

ax.tick_params(labelsize=16)
ax.tick_params(axis='x', rotation=90)

plt.savefig('attention.png')