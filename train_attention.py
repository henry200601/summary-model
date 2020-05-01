import json  
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential,Model,load_model
from tensorflow.keras.layers import Dense, Dropout, Activation, LSTM, TimeDistributed, RepeatVector ,Embedding,Bidirectional,Input,Concatenate,Attention,Lambda,dot,Add
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.losses import BinaryCrossentropy
import tensorflow.keras.backend as K
import tensorflow as tf 
from tensorflow.keras.callbacks import Callback
from tensorflow.keras import backend as K 
from tensorflow.keras.optimizers import RMSprop
from argparse import ArgumentParser 
import pickle

parser=ArgumentParser()
parser.add_argument('--train_data_path')
parser.add_argument('--valid_data_path')
args=parser.parse_args()

MAX_LEN_TEXT=300
MAX_LEN_SUMMARY=40
text_voc_size   =20000

train = pd.read_json(args.train_data_path, lines=True)
valid = pd.read_json(args.valid_data_path, lines=True)

train=train.drop(columns=['sent_bounds','extractive_summary'])
valid=valid.drop(columns=['sent_bounds','extractive_summary'])

train['summary'] = train['summary'].apply(lambda x : '<BOS> '+ x + ' <EOS>')
valid['summary'] = valid['summary'].apply(lambda x : '<BOS> '+ x + ' <EOS>')


text_tokenizer = Tokenizer(num_words=text_voc_size)
text_tokenizer.fit_on_texts(train['text']+' '+train['summary'])




train['text']    =   text_tokenizer.texts_to_sequences(train['text']) 
valid['text']    =   text_tokenizer.texts_to_sequences(valid['text']) 


xtext    =   pad_sequences(train['text'] ,  maxlen=MAX_LEN_TEXT, padding='post', truncating='post') 
ytext    =   pad_sequences(valid['text'] , maxlen=MAX_LEN_TEXT, padding='post',truncating='post')


train['summary']    =   text_tokenizer.texts_to_sequences(train['summary']) 
valid['summary']    =   text_tokenizer.texts_to_sequences(valid['summary']) 


xsum    =   pad_sequences(train['summary'] ,  maxlen=MAX_LEN_SUMMARY, padding='post', truncating='post') 
ysum    =   pad_sequences(valid['summary'] , maxlen=MAX_LEN_SUMMARY, padding='post',truncating='post')


with open("vocabularyattentionindex_word.txt", "wb") as fp:
 pickle.dump(text_tokenizer.index_word, fp)
with open("vocabularyattention.txt", "wb") as fp:
 pickle.dump(text_tokenizer.word_index, fp)

embeddings_index = {}
with open('glove.6B.300d.txt','r') as f:
  
  for line in f:
      values = line.split()
      word = values[0]
      coefs = np.asarray(values[1:], dtype='float32')
      embeddings_index[word] = coefs

embedding_matrix_text = np.zeros((text_voc_size+1, 300))
for word, i in list(text_tokenizer.word_index.items())[:text_voc_size]:
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix_text[i] = embedding_vector

K.clear_session() 
latent_dim = 400 


# Encoder 
encoder_inputs = Input(shape=(MAX_LEN_TEXT,)) 
embedingLayer=Embedding(text_voc_size+1, 
                    300,
                    weights=[embedding_matrix_text],
                    trainable=True)
enc_emb = embedingLayer(encoder_inputs) 

encoder_output, forward_h, forward_c, backward_h, backward_c = Bidirectional(LSTM(latent_dim,return_sequences=True, return_state=True))(enc_emb)
state_h = Concatenate()([forward_h, backward_h])
state_c = Concatenate()([forward_c, backward_c])

encoder_model = Model(inputs=encoder_inputs,outputs=[encoder_output, state_h, state_c]) 





decoder_inputs = Input(shape=(None,))   
dec_emb =embedingLayer(decoder_inputs) 

decoder_outputs,state_h2, state_c2 =LSTM(latent_dim*2, return_sequences=True, return_state=True,dropout=0.6) (dec_emb,initial_state=[state_h , state_c]) 




x = Dense(latent_dim*2, use_bias=False,name='encoder')(encoder_output)

x=dot([decoder_outputs,x],[2,2])
attention_weights=Activation('softmax')(x)
context_vector=dot([attention_weights,encoder_output],[2,1])

decoder_concat_input = Concatenate(axis=-1)([decoder_outputs, context_vector])


decoder_dense=TimeDistributed(Dense(text_voc_size+1, activation='softmax')) 
decoder_outputs =decoder_dense(decoder_concat_input) 


merged_model = Model([encoder_inputs,decoder_inputs], decoder_outputs)
merged_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',metrics=['sparse_categorical_accuracy'])
checkpoint = ModelCheckpoint('newattention05.h5', monitor='val_loss', verbose=1, save_best_only=True,mode='min')
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1,patience=2)
history=merged_model.fit([xtext ,xsum[:,:-1]], xsum.reshape(xsum.shape[0],xsum.shape[1],1)[:,1:] ,epochs=30,callbacks=[checkpoint,es],batch_size=256, validation_data=([ytext ,ysum[:,:-1]], ysum.reshape(ysum.shape[0],ysum.shape[1],1)[:,1:] ))


decoder_state_input_h = Input(shape=(latent_dim*2,))
decoder_state_input_c = Input(shape=(latent_dim*2,))
decoder_hidden_state_input = Input(shape=(MAX_LEN_TEXT,latent_dim*2))


dec_emb2= embedingLayer(decoder_inputs)
decoder_lstm=LSTM(latent_dim*2, return_sequences=True, return_state=True)
decoder_outputs2, state_h2, state_c2 = decoder_lstm(dec_emb2, initial_state=[decoder_state_input_h, decoder_state_input_c])


x = Dense(latent_dim*2, use_bias=False, name='attention_score_vec')(decoder_hidden_state_input)
x=dot([decoder_outputs2,x],[2,2])
attention_weights=Activation('softmax')(x)
context_vector=dot([attention_weights,decoder_hidden_state_input],[2,1])

decoder_inf_concat = Concatenate(axis=-1)([decoder_outputs2, context_vector])
decoder_outputs2 = decoder_dense(decoder_inf_concat)

decoder_model = Model(
[decoder_inputs] + [decoder_hidden_state_input,decoder_state_input_h, decoder_state_input_c],
[decoder_outputs2] + [state_h2, state_c2]+[attention_weights])

for i,j in zip([1,2],[2,3]):
  encoder_model.layers[i].set_weights(merged_model.layers[j].get_weights())
for i,j in zip([1,5,6,11],[2,6,7,12]):
  decoder_model.layers[i].set_weights(merged_model.layers[j].get_weights())

encoder_model.save('attbestencoder01.h5')
decoder_model.save('attbestdecoder01.h5')
