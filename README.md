
# 檔案說明:

## Extractive Summarization
將文章輸入模型，讓模型挑選出文章重要的句字當作摘要
## seq2seq
將文章輸入模型，讓模型自動生成摘要，使用seq2seq
## seq2seq+attention
將文章輸入模型，讓模型自動生成摘要，使用seq2seq+attention


# 使用方法:
## 產生預測檔案 

```
bash ./download.sh  
bash ./extractive.sh TEST_PATH PREDICT_PATH  
bash ./seq2seq.sh TEST_PATH PREDICT_PATH  
bash ./attention.sh TEST_PATH PREDICT_PATH  
```
## 產生訓練模型 

## Extractive Summarization
```
bash ./download.sh (一定要先執行下載，將glove pretrain model 下載至目錄)  
bash ./train.sh TRAIN_PATH VALID_PATH
```
使用後會在目錄下產生:  
檢查點 checkpointjusttest.h5  
訓練模型 passBaseline01.h5  
字典檔 vocabulary.txt

## seq2seq
```
bash ./download.sh (一定要先執行下載，將glove pretrain model 下載至目錄)  
bash ./seq2seqtrain.sh TRAIN_PATH VALID_PATH 
```
使用後會在目錄下產生:   
encoder 模型 seq2seqbestencoder.h5  
decoder 模型 seq2seqbestdecoder.h5
字典檔 vocabularyattentionindex_word.txt ,vocabularyattention.txt

## seq2seq+attention
```
bash ./download.sh (一定要先執行下載，將glove pretrain model 下載至目錄)  
bash ./train_attention.sh TRAIN_PATH VALID_PATH
```
使用後會在目錄下產生:   
encoder 模型 attbestencoder01.h5  
decoder 模型 attbestdecoder01.h5  
字典檔 vocabularyattentionindex_word.txt ,vocabularyattention.txt

# 畫圖

## 第4題
```
bash ./plot_histgram.sh VALID_PATH PREDICT_PATH  
```
目錄下產生成圖片histgram.png
## 第5題
```
bash ./plot_attention.sh VALID_PATH  
```
目錄下產生生成圖片attention.png
