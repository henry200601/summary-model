import json  
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import pickle
from argparse import ArgumentParser 

parser=ArgumentParser()
parser.add_argument('--test_data_path')
parser.add_argument('--output_path')
args=parser.parse_args()

def total_sent(test):
  total=0
  for i in range(len(test)):
    total+=len(test['sent_bounds'][i])
  return total

def create_newtest(total,test):
  newtest=pd.DataFrame(pd.np.empty((total,2))*np.nan, columns = ['id','sentence'])
  newtest['sentence']=''

  i=0
  for index, row  in test.iterrows():
    for ind,(l,r) in enumerate(row['sent_bounds']):
        newtest.at[i,'id']=row['id']
        newtest.at[i,'sentence']=row['text'][l:r]
        i+=1
  newtest['id']=newtest['id'].astype(int)
  return newtest

def weight_BCE(y_true, y_pred):
  y_true=tf.cast(y_true,tf.float32)
  y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
  return tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=tf.math.log(y_pred / (1 - y_pred)), labels=y_true, pos_weight=13))

def preprocesing(test):
  unpading_sentences=[]
  sent_bounds=[]
  for ind_train,id_train in enumerate(test['id']):
    sen=[]
    sent_bound=[]
    r=0
    l=0
    for i,row in newtest[newtest['id']==id_train].iterrows():
      sen+=row['sentence']
      l+=len(row['sentence'])
      sent_bound.append([r,l])
      r=l
    sent_bounds.append(sent_bound)
    unpading_sentences.append(sen)
  xtrain=pad_sequences(unpading_sentences, maxlen=329,padding='post',truncating='post')
  return xtrain,sent_bounds
"""## test"""

test = pd.read_json(args.test_data_path, lines=True)

total=total_sent(test)
newtest=create_newtest(total,test)

#tokenize
with open('vocabulary.txt','rb') as fp:
  vocabulary=pickle.load(fp)
tokenizer=Tokenizer(oov_token='<UNK>')
tokenizer.word_index = vocabulary.copy()
newtest['sentence']=tokenizer.texts_to_sequences(newtest['sentence'])

#preprocessing
xtrain,sent_bounds=preprocesing(test)

#predict
model=load_model('passBaseline01.h5',custom_objects={'weight_BCE': weight_BCE})
xpred=model.predict(xtrain,batch_size=256)
pred=np.squeeze(xpred, axis=-1)
pred=np.where(pred >=0.5,1,0)

#postprocessing
predict_sentence_index=[]
for item,bounds in zip(pred.tolist(),sent_bounds):
  sentence_index=[]
  for ind,(l,r) in enumerate(bounds):
    if item[l:r].count(1)>=(r-l):
      sentence_index.append(ind)
  predict_sentence_index.append(sentence_index)
  
test['predict_sentence_index']=''
for i in range(len(test)):
  test.at[i,'predict_sentence_index']=predict_sentence_index[i]

test=test.filter(items=['id','predict_sentence_index'])
test['id']=test['id'].apply(str)


test.to_json(args.output_path,orient='records',lines=True)