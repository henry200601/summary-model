import json  
import pandas as pd
import nltk
import numpy as np
import pickle

 
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, LSTM, TimeDistributed, RepeatVector ,Embedding,Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.losses import BinaryCrossentropy
import tensorflow.keras.backend as K
import tensorflow as tf

from tensorflow.keras.callbacks import Callback

from argparse import ArgumentParser 

parser=ArgumentParser()
parser.add_argument('--train_data_path')
parser.add_argument('--valid_data_path')
args=parser.parse_args()



"""## 讀取檔案"""

train = pd.read_json(args.train_data_path, lines=True)
valid = pd.read_json(args.valid_data_path, lines=True)

"""## trian preprocessing"""
def create_sentences(train):
  total=0
  for i in range(len(train)):
    total+=len(train['sent_bounds'][i])
  
  newtrain=pd.DataFrame(pd.np.empty((total,3))*np.nan, columns = ['id','sentence','label'])
  newtrain['sentence']=''

  i=0
  for index, row  in train.iterrows():
    for ind,(l,r) in enumerate(row['sent_bounds']):
        newtrain.at[i,'id']=row['id']
        newtrain.at[i,'sentence']=row['text'][l:r]
        newtrain.at[i,'label']=1 if ind==row['extractive_summary'] else 0
        i+=1
  newtrain['id']=newtrain['id'].astype(int)
  newtrain['label']=newtrain['label'].astype(int)
  return newtrain

def weight_BCE(y_true, y_pred):
  y_true=tf.cast(y_true,tf.float32)
  y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
  return tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=tf.math.log(y_pred / (1 - y_pred)), labels=y_true, pos_weight=13))

def create_new_sentbounds(train,newtrain):
  unpading_sentences=[]
  unpading_label=[]
  sent_bounds=[]
  for ind_train,id_train in enumerate(train['id']):
    temp=[]
    sen=[]
    sent_bound=[]
    r=0
    l=0
    for i,row in newtrain[newtrain['id']==id_train].iterrows():
      sen+=row['sentence']
      l+=len(row['sentence'])
      if row['label']==1:
        temp+=len(row['sentence'])*[1]
      else:
        temp+=len(row['sentence'])*[0]

      sent_bound.append([r,l])
      r=l
    sent_bounds.append(sent_bound)
    unpading_sentences.append(sen)
    unpading_label.append(temp)
  train=pad_sequences(unpading_sentences, maxlen=329,padding='post',truncating='post')
  label=pad_sequences(unpading_label, maxlen=329,padding='post',truncating='post')
  return train,label,sent_bounds


# newtrain.to_csv('/content/drive/My Drive/mui-lab/hw1/train.csv',index=0)
"""## 將摘要分句"""
newtrain=create_sentences(train)
newvalid=create_sentences(valid)

"""## 分詞"""

tokenizer=Tokenizer(oov_token="<UNK>")
tokenizer.fit_on_texts(newtrain['sentence'])
newtrain['sentence']=tokenizer.texts_to_sequences(newtrain['sentence'])
newvalid['sentence']=tokenizer.texts_to_sequences(newvalid['sentence'])

max_index=max(tokenizer.word_index.values())+1

with open("vocabulary.txt", "wb") as fp:
 pickle.dump(tokenizer.word_index, fp, pickle.HIGHEST_PROTOCOL)



xtrain,xlabel,xsent_bounds=create_new_sentbounds(train,newtrain)
ytrain,ylabel,ysent_bounds=create_new_sentbounds(valid,newvalid)


#pretrain embeding weights
embeddings_index = {}
with open('glove.6B.300d.txt','r') as f:#路徑記得改
  
  for line in f:
      values = line.split()
      word = values[0]
      coefs = np.asarray(values[1:], dtype='float32')
      embeddings_index[word] = coefs

embedding_matrix = np.zeros((len(tokenizer.word_index)+1, 300))
for word, i in tokenizer.word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
"""## Loss function"""


"""## successful code"""

K.clear_session()
model=Sequential()
model.add(Embedding(146884,
                            300,
                            weights=[embedding_matrix],
                            input_length=329,
                          
                            trainable=False))
# model.add(Dropout(0.1))

model.add(Bidirectional(LSTM(600,return_sequences=True)))
# model.add(Bidirectional(LSTM(100,return_sequences=True, recurrent_dropout=0.1)))
#model.add(Bidirectional(LSTM(64,return_sequences=True)))
model.add(TimeDistributed(Dense(100,activation='relu')))
model.add(TimeDistributed(Dense(50,activation='relu')))
model.add(TimeDistributed(Dense(50,activation='relu'))) 
model.add(TimeDistributed(Dense(1,activation='sigmoid')))    # or use model.add(Dense(1))

model.compile(loss=weight_BCE, optimizer="adam",metrics=['acc'])
model.summary()

checkpoint = ModelCheckpoint('checkpoint.h5', monitor='val_loss', verbose=1, save_best_only=True,
mode='min')

callbacks_list = [checkpoint]


history=model.fit(xtrain, xlabel,
          batch_size=256,
          epochs=10,
          callbacks=callbacks_list,
          validation_data=(ytrain, ylabel))


model.save('passBaseline01.h5')


