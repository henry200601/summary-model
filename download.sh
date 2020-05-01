#!/usr/bin/env bash
glove=https://www.dropbox.com/s/5z7a4fje295bzsy/glove.6B.300d.txt?dl=1
model=https://www.dropbox.com/s/qxuxhpp8zune5om/passBaseline01.h5?dl=1

https://www.dropbox.com/s/mk0g1issf15aosw/seq2seqbestdecoder.h5?dl=1

wget  "${glove}"  -O ./glove.6B.300d.txt
wget  "${model}"  -O ./passBaseline01.h5
wget https://www.dropbox.com/s/e405r3mnlizhs9o/vocabulary.txt?dl=1 -O./vocabulary.txt


wget https://www.dropbox.com/s/mk0g1issf15aosw/seq2seqbestdecoder.h5?dl=1 -O./seq2seqbestdecoder.h5
wget https://www.dropbox.com/s/jhm7h0pp5rqetl1/seq2seqbestencoder.h5?dl=1 -O./seq2seqbestencoder.h5

wget https://www.dropbox.com/s/yjanl7vds775lc9/attbestdecoder01.h5?dl=1 -O./attbestdecoder01.h5
wget https://www.dropbox.com/s/00suu87uvuj4p4p/attbestencoder01.h5?dl=1 -O./attbestencoder01.h5


wget https://www.dropbox.com/s/4bggp4bq256rpgo/vocabularyattention.txt?dl=1 -O./vocabularyattention.txt
wget https://www.dropbox.com/s/jk44n5bq4knd934/vocabularyattentionindex_word%20.txt?dl=1 -O./vocabularyattentionindex_word.txt
