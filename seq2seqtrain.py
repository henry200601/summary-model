import json  
import pandas as pd
import nltk
import numpy as np
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.models import Sequential,Model,load_model
from tensorflow.keras.layers import Dense, Dropout, Activation, LSTM, TimeDistributed, RepeatVector ,Embedding,Bidirectional,Input,Concatenate,Lambda
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.losses import BinaryCrossentropy
import tensorflow.keras.backend as K
import tensorflow as tf
   
from tensorflow.keras.callbacks import Callback
from tensorflow.keras import backend as K 


from argparse import ArgumentParser 

parser=ArgumentParser()
parser.add_argument('--train_data_path')
parser.add_argument('--valid_data_path')
args=parser.parse_args()

train = pd.read_json(args.train_data_path, lines=True)
valid = pd.read_json(args.valid_data_path, lines=True)


train=train.drop(columns=['sent_bounds','extractive_summary'])
valid=valid.drop(columns=['sent_bounds','extractive_summary'])

train['summary'] = train['summary'].apply(lambda x : '<BOS> '+ x + ' <EOS>')
valid['summary'] = valid['summary'].apply(lambda x : '<BOS> '+ x + ' <EOS>')

MAX_LEN_TEXT=300
MAX_LEN_SUMMARY=40
text_voc_size   =20000


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
        # words not found in embedding index will be all-zeros.
        embedding_matrix_text[i] = embedding_vector
        
K.clear_session() 
latent_dim = 300 

encoder_inputs = Input(shape=(MAX_LEN_TEXT,)) 
embedingLayer=Embedding(text_voc_size+1, 
                    latent_dim,
                    weights=[embedding_matrix_text],
                    trainable=True)
enc_emb = embedingLayer(encoder_inputs) 
encoder_output, forward_h, forward_c, backward_h, backward_c = Bidirectional(LSTM(latent_dim, return_state=True),merge_mode='ave')(enc_emb)
state_h = Concatenate()([forward_h, backward_h])
state_c = Concatenate()([forward_c, backward_c])
state_h = Dense(latent_dim,'tanh')(state_h)
state_c = Dense(latent_dim,'tanh')(state_c)

encoder_model = Model(inputs=encoder_inputs,outputs=[encoder_output, state_h, state_c]) 

# decoder model
decoder_inputs = Input(shape=(None,))   
dec_emb =embedingLayer(decoder_inputs) 

decoder_outputs,state_h2, state_c2 =LSTM(latent_dim, return_sequences=True, return_state=True) (dec_emb,initial_state=[state_h , state_c]) 

decoder_outputs = TimeDistributed(Dense(text_voc_size+1, activation='softmax')) (decoder_outputs) 


merged_model = Model([encoder_inputs,decoder_inputs], decoder_outputs)
merged_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',metrics=['acc'])
checkpoint = ModelCheckpoint('retrainhw2.h5', monitor='val_loss', verbose=1, save_best_only=True,mode='min')
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)
history=merged_model.fit([xtext ,xsum[:,:-1]], xsum.reshape(xsum.shape[0],xsum.shape[1],1)[:,1:] ,epochs=30,callbacks=[checkpoint,es],batch_size=256, validation_data=([ytext ,ysum[:,:-1]], ysum.reshape(ysum.shape[0],ysum.shape[1],1)[:,1:] ))


# encoder_model.save('encoder.h5')

decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))



dec_emb2= embedingLayer(decoder_inputs)
decoder_lstm=LSTM(latent_dim, return_sequences=True, return_state=True)

decoder_outputs2, state_h2, state_c2 = decoder_lstm(dec_emb2, initial_state=[decoder_state_input_h, decoder_state_input_c])

decoder_dense=TimeDistributed(Dense(text_voc_size+1, activation='softmax')) 
decoder_outputs2 = decoder_dense(decoder_outputs2)

decoder_model = Model(
[decoder_inputs] + [decoder_state_input_h, decoder_state_input_c],
[decoder_outputs2] + [state_h2, state_c2])

for i,j in zip([1,4,5],[2,8,9]):
  decoder_model.layers[i].set_weights(merged_model.layers[j].get_weights())

for i,j in zip([1,2,5,6],[2,3,6,7]):
  encoder_model.layers[i].set_weights(merged_model.layers[j].get_weights())

decoder_model.save('seq2seqbestdecoder.h5')
encoder_model.save('seq2seqbestencoder.h5')
