#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 25 11:03:27 2017

@author: Satyarth Vaidya
"""

import gensim
import nltk.data
from nltk.corpus import stopwords
import argparse
import os
import re
import logging
import sys
from gensim.models.keyedvectors import KeyedVectors
import pandas as pd

sentence_detector = nltk.data.load('tokenizers/punkt/german.pickle')
punctuation_tokens = ['.', '..', '...', ',', ';', ':', '(', ')', '"', '“',',','„','\'', '[', ']', '{', '}', '?', '!', '-', u'–', '+', '*', '--', '\'\'', '``']
punctuation = '?.!/;:()&+„.“'

def replace_umlauts(text):
    res = text
    res = res.replace(u'ä', 'ae')
    res = res.replace(u'ö', 'oe')
    res = res.replace(u'ü', 'ue')
    res = res.replace(u'Ä', 'Ae')
    res = res.replace(u'Ö', 'Oe')
    res = res.replace(u'Ü', 'Ue')
    res = res.replace(u'ß', 'ss')
    return res

# get stopwords
umlauts = True
#model = gensim.models.KeyedVectors.load_word2vec_format('/home/Downloads/DB_POC_II/data_web_download/german.model', binary=True)
model = gensim.models.KeyedVectors.load_word2vec_format('/home/Desktop/Downloads/DB_POC_II/initial_finance_model/model/fin_data.model', binary=True)



data = '/home/Desktop/Downloads/DB_POC_II/data_web_download/data/economy.articles'
target = '/home/Desktop/Downloads/DB_POC_II/data_web_download/data/economy_result.articles'

stop_words = stopwords.words('german') if not umlauts == True else [replace_umlauts(token) for token in stopwords.words('german')]
num_sentences = sum(1 for line in open(data))


output = open(target , 'w')
i = 0
with open(data, 'r') as infile:
    for line in infile:
        i += 1
        sentences = sentence_detector.tokenize(line)
        for sentence in sentences:
            #print (sentence)
            sentence = replace_umlauts(sentence)
            words = nltk.word_tokenize(sentence)
            words = [x for x in words if x not in punctuation_tokens]
            words = [re.sub('[' + punctuation + ']', '', x) for x in words]
            words = [x for x in words if x not in stop_words]
            if len(words)>1:
                #print (words)
                output.write(' '.join(words) + ' ')
        output.write('\n\n')
        
        


ct = 0
row_list = []
titles = ['Anlagestrategie' , 'Aktienkurse', 'Wirtschaft']

df = pd.DataFrame(columns=['Title', 'Similarity Score'])
with open(target , 'r') as processd_file:
    for line in processd_file:
        ct +=1
        sentences = sentence_detector.tokenize(line)
        for sentence in sentences:
            words = sentence.split()
            for category in titles:
                dict1 = {}
                num_word_sentence = 0
                num_score = 0
                for word in words:
                    if word in model.index2word:
                        x = model.similarity(word , category)
                        num_score+=x
                        num_word_sentence+=1
                dict1.update({'Title': category , 'Similarity Score': round(num_score/num_word_sentence,3)})
                row_list.append(dict1)

        
df = pd.DataFrame(row_list)               
df.to_csv('/home/Downloads/DB_POC_II/german_code_e2e/Economy_score.csv', index = False)    
               
                
            
            
        
 
