#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 16:23:24 2017

@author: Satyarth Vaidya
"""
import pandas as pd
import gensim
import nltk.data
from nltk.corpus import stopwords
import re
import random

from sklearn import metrics

sentence_detector = nltk.data.load('tokenizers/punkt/german.pickle')
punctuation_tokens = [ '\n','/',"'—" ,' #'  ,'”',  '·',' . ','’', '‘','%','.','#','@' ,"'—",'|','«','» |','|','»','.', '..', '...', ',', ';', ':', '(', ')', '"', '“',',','„','\'', '[', ']', '{', '}', '?', '!', '-', u'–', '+', '*', '--', '\'\'', '``']
punctuation = "|?.!/;:()&+„.“|0-9'— ·#\r\n\r\n\r\n\xa0-\xa0\r\n\r\n/"

Yahoo_data = pd.read_csv(r'/home/Downloads/German Data/Yahoo_Data.csv')

Handelsblatt_data = pd.read_csv(r'/home/Downloads/German Data/Handelsblatt.csv')
Handelsblatt_data = Handelsblatt_data.drop('Unnamed: 0' , 1)

Blog_data_1 = pd.read_csv(r'/home/Downloads/German Data/Final_Blog2.csv')
Blog_data_1 = Blog_data_1.drop('Unnamed: 0' , 1)

Blog_data_2 = pd.read_csv(r'/home/Downloads/German Data/Comments_Blog.csv')
Blog_data_2 = Blog_data_2.drop('Unnamed: 0' , 1)

Taz_data = pd.read_csv('/home/Downloads/German Data/taz.csv')

Finance_data = pd.concat([Yahoo_data , Handelsblatt_data, Blog_data_1, Blog_data_2,Taz_data] )

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


Finance_data['Article_processed'] = Finance_data['Article'].apply(lambda x: x.lower())
Finance_data['Article_processed'] = Finance_data['Article_processed'].str.replace(r'\d+', '')
Finance_data['Article_processed'] = Finance_data['Article_processed'].apply(replace_umlauts)

Finance_data['Tokenized'] = Finance_data['Article_processed'].apply(nltk.word_tokenize)
Finance_data['Tokenized'] = Finance_data['Tokenized'].apply(lambda x : [item for item in x if item not in punctuation_tokens])
Finance_data['Tokenized'] = Finance_data['Tokenized'].apply(lambda x : [ re.sub('[' + punctuation + ']', '', item) for item in x])
Finance_data['Tokenized'] = Finance_data['Tokenized'].apply(lambda x : [item for item in x if item not in stop_words])

sentences  = Finance_data.Tokenized.tolist()ValueError: could not convert string to float: 'apples'


## Word2Vec Model
finance_model = gensim.models.Word2Vec(sentences,  seed = 0 , size=150, window=10, min_count=20, workers=4)
finance_model_v2 = gensim.models.Word2Vec(sentences,  seed = 0 , size=150, window=10, min_count=10, workers=4)
vocabulary = list(finance_model.wv.vocab.keys())
finance_model.wv.save_word2vec_format('finance.model', binary=True)


#Doc2vec model training
sentences_ = [gensim.models.doc2vec.LabeledSentence(words_lis, ['SENT_%d'%index]) \
              for index,words_lis in enumerate(list(Finance_data['Tokenized'].ravel()))]

d2v_model = gensim.models.Doc2Vec(size=100,alpha=0.025,min_alpha=0.025,window=8,min_count=5,seed=1,workers=4)
d2v_model.build_vocab(sentences_)

for epoch in range(10):
    print('Epoch %d'%epoch)
    random.shuffle(sentences_)
    #d2v_model.train(sentences_ , total_examples = d2v_model.corpus_count, epochs = d2v_model.iter)
    d2v_model.train(sentences_)
    d2v_model.alpha-=0.002
    d2v_model.min_alpha = d2v_model.alpha


Finance_data['dvecs'] = [d2v_model.docvecs['SENT_%d'%i] for i in range(Finance_data.shape[0])]

##Creating new distance vectors

Finance_data['economy_distance']  =  [1 - abs(metrics.pairwise.cosine_similarity(Finance_data.iloc[i,5],d2v_model['wirtschaft'])) for i in range(Finance_data.shape[0]) ]
Finance_data['brexit_distance']  =  [1 - abs(metrics.pairwise.cosine_similarity(Finance_data.iloc[i,5],d2v_model['brexit'])) for i in range(Finance_data.shape[0]) ]
Finance_data['investment_distance']  =  [1 - abs (metrics.pairwise.cosine_similarity(Finance_data.iloc[i,5],d2v_model['investition'])) for i in range(Finance_data.shape[0]) ]
Finance_data['max_score']  = Finance_data[['brexit_distance' , 'economy_distance' , 'investment_distance' ]].max(axis=1)

Finance_data = Finance_data.sort_values(['max_score'] , ascending = False)
Finance_data[0:20].to_csv("Top_20_articles.csv", index = False)
Finance_data[-20:].to_csv("Last_20_articles.csv", index = False)
#titles = ['wirtschaft' , 'brexit' , 'investition']
