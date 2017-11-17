#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 14:14:41 2017

@author: Satyarth Vaidya
"""
import pandas as pd
import gensim
import nltk.data
from nltk.corpus import stopwords
import re
import random
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn import metrics
import numpy as np
from sklearn.decomposition import PCA

sentence_detector = nltk.data.load('tokenizers/punkt/german.pickle')
punctuation_tokens = [ '\n','/',"'—" ,' #'  ,'”',  '·',' . ','’', '‘','%','.','#','@' ,"'—",'|','«','» |','|','»','.', '..', '...', ',', ';', ':', '(', ')', '"', '“',',','„','\'', '[', ']', '{', '}', '?', '!', '-', u'–', '+', '*', '--', '\'\'', '``']
punctuation = "|?.!/;:()&+„.“|0-9'— ·#\r\n\r\n\r\n\xa0-\xa0\r\n\r\n/"

Yahoo_data = pd.read_csv(r'/home/fractaluser/Downloads/German Data/Yahoo_Data.csv')
Yahoo_data = Yahoo_data.drop_duplicates(['Article'])

Handelsblatt_data = pd.read_csv(r'/home/fractaluser/Downloads/German Data/Handelsblatt.csv')
Handelsblatt_data = Handelsblatt_data.drop('Unnamed: 0' , 1)  
Handelsblatt_data = Handelsblatt_data.drop_duplicates(['Article'])

Blog_data_1 = pd.read_csv(r'/home/fractaluser/Downloads/German Data/Final_Blog2.csv')
Blog_data_1 = Blog_data_1.drop('Unnamed: 0' , 1)
Blog_data_1 = Blog_data_1.drop_duplicates(['Article'])

Blog_data_2 = pd.read_csv(r'/home/fractaluser/Downloads/German Data/Comments_Blog.csv')
Blog_data_2 = Blog_data_2.drop('Unnamed: 0' , 1)
Blog_data_2 = Blog_data_2.drop_duplicates(['Article'])

Taz_data = pd.read_csv('/home/fractaluser/Downloads/German Data/taz.csv')
Taz_data = Taz_data.drop('Unnamed: 0' , 1)
Taz_data = Taz_data.drop_duplicates(['Article'])

Finance_data = pd.concat([Yahoo_data , Handelsblatt_data, Blog_data_1, Blog_data_2 , Taz_data] )

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

stop_words = stopwords.words('german') 
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

sentences  = Finance_data.Tokenized.tolist()

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
new_data = Finance_data['dvecs'] 
transformed_data = np.vstack(new_data)
reduced_data = PCA(n_components=2).fit_transform(transformed_data)

## Apply K Means Clustering on the Data
clusters=3
km = KMeans(n_clusters=clusters , random_state  = 0 ,  init='k-means++')
km.fit(reduced_data)

# Step size of the mesh. Decrease to increase the quality of the VQ.
h = .02     # point in the mesh [x_min, x_max]x[y_min, y_max].

# Plot the decision boundary. For that, we will assign a color to each
x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Obtain labels for each point in mesh. Use last trained model.
Z = km.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(1)
plt.clf()
plt.imshow(Z, interpolation='nearest',
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap=plt.cm.Paired,
           aspect='auto', origin='lower')

plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
# Plot the centroids as a white X
centroids = km.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1],
            marker='x', s=169, linewidths=3,
            color='w', zorder=10)
plt.title('K-means clustering on the Document Vectors (PCA-reduced data)')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.show()


labels = km.labels_

    
for i in range(0, clusters):
    data_label = Finance_data.iloc[labels==i]
    data_label.to_csv('data_table_'+ str(i)+'.csv' , index = False)
    
    
#from sklearn.feature_extraction.text import TfidfVectorizer
    
#vectorizer = TfidfVectorizer(use_idf = True,stop_words = stopset,ngram_range = {1,3} ,lowercase = True)
#X = vectorizer.fit_transform(Finance_data['Article'])
