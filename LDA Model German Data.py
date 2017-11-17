#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 12:07:56 2017

@author: Satyarth Vaidya
"""

import pandas as pd
import gensim
import nltk.data
from nltk.corpus import stopwords
import re
import os

from gensim import corpora, models

#Finance_data = pd.concat([Yahoo_data , Handelsblatt_data, Blog_data_1, Blog_data_2 , Taz_data] )

data_dir = '/Datadrive/German_finance/fin_specific'
files = os.listdir(data_dir)
files = [os.path.join(data_dir,file) for file in files]
punctuation_tokens = [ '\n','/',"'—" ,' #'  ,'”',  '·',' . ','’', '‘','%','.','#','@' ,"'—",'|','«','» |','|','»','.', '..', '...', ',', ';', ':', '(', ')', '"', '“',',','„','\'', '[', ']', '{', '}', '?', '!', '-', u'–', '+', '*', '--', '\'\'', '``']
punctuation = "|?.!/;:()&+„.“|0-9'— ·#\r\n\r\n\r\n\xa0-\xa0\r\n\r\n/"

    
print(files)
data = []
cnt =0
for file in files:
    f = open(file)
    for line in f:
        cnt+=1
        if cnt%1000000 == 0:
            print('processed {0} lines from {1}'.format(cnt, file))
        data+= [line]
        
Finance_data = pd.DataFrame(data , columns = ['Article'])        


stop_words = stopwords.words('german') 


def replace_umlauts(text):
    res = text
    res = res.replace(u'ä', 'ae')
    res = res.replace(u'ö', 'oe')
    res = res.replace(u'ü', 'ue')
    res = res.replace(u'Ä', 'Ae')
    res = res.replace(u'Ö', 'Oe')
    res = res.replace(u'Ü', 'Ue')
    res = res.replace(u'ß', 'ss')
    res = res.replace('-', ' ')
    return res



stop_words_new = []
for txt in stop_words:
    stop_words_new.append(replace_umlauts(txt))

Finance_data['Article_processed'] = Finance_data['Article'].apply(lambda x: x.lower())
Finance_data['Article_processed'] = Finance_data['Article_processed'].str.replace(r'\d+', '')
Finance_data['Article_processed'] = Finance_data['Article_processed'].apply(replace_umlauts)

Finance_data['Tokenized'] = Finance_data['Article_processed'].apply(nltk.word_tokenize)
Finance_data['Tokenized'] = Finance_data['Tokenized'].apply(lambda x : [item for item in x if item not in punctuation_tokens])
Finance_data['Tokenized'] = Finance_data['Tokenized'].apply(lambda x : [ re.sub('[' + punctuation + ']', '', item) for item in x])
Finance_data['Tokenized'] = Finance_data['Tokenized'].apply(lambda x : [item for item in x if item not in stop_words_new])

texts  = Finance_data.Tokenized.tolist()
dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=3, id2word = dictionary, passes=20)
print(ldamodel.print_topics(num_topics=3, num_words=3))
