
# 檔案說明:
## Extractive Summarization
download.sh 下載所需檔案  
train.sh 執行train.py的檔案  
train.py extractive的訓練檔案  
extractive.sh 為產生Extractive Summarization的預測檔案  
early.sh 執行early.py的檔案  
early.py 為產生預測的檔案

## seq2seq

seq2seq.sh seq2seq.py 的執行檔案  
seq2seq.py seq2seq的預測檔案  
seq2seqtrain.py seq2seq的訓練檔案  
plot_histgram.sh 畫出直方圖的檔案  
plot_histgram.py 畫出直方圖的檔案

## seq2seq+attention

attention.sh 的執行檔案  
attention.py attention的預測檔案  
train_attention.py  seq2seq+attention的訓練檔  
train_attention.sh  seq2seq+attention的訓練檔  
plot_attention.sh 畫出attention weights的檔案    
plot_attention.py 畫出attention weights的檔案   

# 使用方法:
## 產生預測檔案 
bash ./download.sh  
bash ./extractive.sh TEST_PATH PREDICT_PATH  
bash ./seq2seq.sh TEST_PATH PREDICT_PATH  
bash ./attention.sh TEST_PATH PREDICT_PATH  

## 產生訓練模型 

## Extractive Summarization
bash ./download.sh (一定要先執行下載，將glove pretrain model 下載至目錄)  
bash ./train.sh TRAIN_PATH VALID_PATH

使用後會在目錄下產生:  
檢查點 checkpointjusttest.h5  
訓練模型 passBaseline01.h5  
字典檔 vocabulary.txt

## seq2seq
bash ./download.sh (一定要先執行下載，將glove pretrain model 下載至目錄)  
bash ./seq2seqtrain.sh TRAIN_PATH VALID_PATH 

使用後會在目錄下產生:   
encoder 模型 seq2seqbestencoder.h5  
decoder 模型 seq2seqbestdecoder.h5
字典檔 vocabularyattentionindex_word.txt ,vocabularyattention.txt

## seq2seq+attention
bash ./download.sh (一定要先執行下載，將glove pretrain model 下載至目錄)  
bash ./train_attention.sh TRAIN_PATH VALID_PATH

使用後會在目錄下產生:   
encoder 模型 attbestencoder01.h5  
decoder 模型 attbestdecoder01.h5  
字典檔 vocabularyattentionindex_word.txt ,vocabularyattention.txt

# 畫圖

## 第4題
bash ./plot_histgram.sh VALID_PATH PREDICT_PATH  
目錄下產生成圖片histgram.png
## 第5題
bash ./plot_attention.sh VALID_PATH  
目錄下產生生成圖片attention.png