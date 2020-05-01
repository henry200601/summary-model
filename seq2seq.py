from tensorflow.keras.models import load_model
from tqdm import tqdm
import pickle
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from argparse import ArgumentParser 
parser=ArgumentParser()
parser.add_argument('--test_data_path')
parser.add_argument('--output_path')
args=parser.parse_args()

test = pd.read_json(args.test_data_path, lines=True)
test=test.filter(items=['id','text'])

with open("vocabularyattention.txt", "rb") as fp:
 vocabulary=pickle.load(fp)
with open("vocabularyattentionindex_word.txt", "rb") as fp:
 index_word=pickle.load(fp)

encoder_model=load_model('seq2seqbestencoder.h5')
decoder_model=load_model('seq2seqbestdecoder.h5')

text_tokenizer=Tokenizer(num_words=20000)
text_tokenizer.word_index = vocabulary.copy()
test['text']=text_tokenizer.texts_to_sequences(test['text'])
ytext=pad_sequences(test['text'],  maxlen=300, padding='post', truncating='post') 
test['predict']=''#把每句長度紀錄下來
res=np.zeros((len(test),20))

for i in tqdm(range(100)):  
  e_out, e_h, e_c = encoder_model.predict(ytext[200*i:200*(i+1)],batch_size=256)
  target_seq = np.zeros((200,1))
  target_seq[:, 0] = text_tokenizer.word_index['bos']

  for j in range(20):
    output_tokens, h, c = decoder_model.predict([target_seq[:,-1]] + [e_h, e_c],batch_size=256)
    sampled_token_index = np.argmax(output_tokens,axis=-1)
    res[200*i:200*(i+1),j]=target_seq[:,0]
    target_seq = np.zeros((200,1))
    target_seq[:, 0] = sampled_token_index[:,0]
    e_h, e_c=h,c
    
test['predict']=''
test=test.drop(columns=['text'])
for i,sent in enumerate(res[:,1:].tolist()):
  decoded_sentence=[]
  for word in sent:
    if word == 0:
      break
    sampled_token = index_word[word]
    if (sampled_token=='eos'):
      break
    decoded_sentence.append(sampled_token)
  test.at[i,'predict']=' '.join(decoded_sentence)
test['id'].apply(str)
test['id']=test['id'].apply(str)
test.to_json(args.output_path,orient='records',lines=True)