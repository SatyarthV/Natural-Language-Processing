k#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 16:05:02 2017

@author: Satyarth Vaidya
"""

import os
import re

from numpy.random import shuffle

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from gensim import models

import pandas as pd

from nltk.stem import SnowballStemmer

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
    

stemmer = SnowballStemmer("german")

stop_words = stopwords.words('german') 
files_path = '_____'

stop_words_new = []
for word in stop_words:
    stop_words_new.append(replace_umlauts(word))
    
files = [os.path.join(files_path, file) for file in os.listdir(files_path) if file.endswith('.csv')]

Finance_data = pd.concat(list(map(pd.read_csv, files)), ignore_index = True)

Finance_data['Article_processed'] = Finance_data['0'].str.lower()
Finance_data['Article_processed'] = Finance_data['Article_processed'].apply(replace_umlauts)
Finance_data['Article_processed'] = Finance_data['Article_processed'].apply(lambda ln: re.compile("[^a-z]").sub(" ", ln))

Finance_data['Tokenized'] = Finance_data['Article_processed'].str.split()

Finance_data['Tokenized_clean'] = Finance_data['Tokenized'].apply(lambda tokens: [stemmer.stem(token) for token in tokens if token not in stop_words_new and len(token)>1])

sentences = Finance_data.Tokenized_clean.values

w2v_model = models.Word2Vec(sentences, seed=0, size = 300, window=5, min_count = 5, workers = 10)

#Creating a doc2vec model here
sentences = [models.doc2vec.LabeledSentence(words_lis, ['SENT_%d' %index]) for index, words_lis in enumerate(sentences)]

d2v_model = models.Doc2Vec(size = 300, alpha = 0.025, min_alpha = 0.025,window=5, min_count = 5, seed=0, workers = 10)
d2v_model.build_vocab(sentences)

for eopch in range(1, 10):
    print('Epoch {0}'.format(eopch))
    shuffle(sentences)
    d2v_model.train(sentences)
    d2v_model.alpha-=0.002
    d2v_model.min_alpha = d2v_model.alpha
    
    
#Evaluating the topics here
topic_buckets = {'reports': ['bericht', 'meldung', 
                             'report', 'gutachten', 
                             'protokoll', 'reportage',
                             'mitteilung', 'rapport',
                             'zeugnis', 'anzeige', 'liste',
                             'assuage', 'verlautbarung', 'zensuren'],
                 'banking': ['banking', 'bankwesen', 'schraeglage'],
                 'assets':['vermoegenswerte', 'aktiva'],
                 'investment': ['investition', 'wertpapier', 'investierung',
                                'belagerung', 'einsetzung', 'amtseinsetzung'],
                 'cash': ['kasse', 'bargeld', 'kleingeld', 'kassenbestand',
                          'geldmittel'],
                 'liability': ['haftung', 'haftpflicht', 'leistungspflicht',
                               'passiva', 'schulden']}
                 
