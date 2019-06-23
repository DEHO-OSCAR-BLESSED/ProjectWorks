import nltk
import  re
from nltk.corpus import stopwords
from nltk import *
from nltk.cluster import KMeansClusterer
from pprint import pprint
import  numpy as np
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from matplotlib import pyplot

StopWords= stopwords.words('english')

Data= 'During inspection it was found that the POS1 ' \
      'strut is leaking and requires replacement Raise notification :' \
      ' 2 x fitters * Machine will be down * 28.07.2018 11:06:18 UTC Damon Stacey (STACDA9) ' \
      'THIS JOB HAS BEEN RESCHEDULED 14.09.2018 03:24:15 UTC Scott Williams (WILLSO9) Advised by site that hub i' \
      's also leaking now Last time this corner was off a new strut was fitted but a second hand hub was all that was available at the time.' \
      ' I have added the material number for a complete as the hub and strut needs to be changed. Komatsu have been engaged ' \
      'for warranty on the strut. See attached email for photos from site. 16.10.2018 10:50:13 UTC Traven Hooper (HOOPT9)' \
      ' Job has been completed 16.10.2018 20:49:54 UTC Gavin Casten (CASTG91) Phone +6149805626 Work completed ahead of schedule'

#--------------Cleaning data-----------------------------


Cleaned= []
for word in Data.split(' '):
    if word not in StopWords and re.match('[a-zA-Z\-][a-zA-Z\-]{2,}', word):
        Cleaned.append(word)
#print(Cleaned)

#--------------------------------------Word2Vec Model For Feature Extraction---------------------------------------------------------------
# Trainning Word2Vec Model

sentence = [[sent] for sent in Cleaned]
model = Word2Vec(sentence,  min_count=1 , sg=1,size=300, workers=3)
#model.save("Word2Vec Model")

#Fitting  a 2d PCA model to the vectors
X = model[model.wv.vocab]
pca = PCA(n_components=2)
result = pca.fit_transform(X)
pyplot.scatter(result[:, 0], result[:, 1])
words = list(model.wv.vocab)
#print(words)
for i, word in enumerate(words):
    pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))

pyplot.show()


#-------------------------------Feeding Word Embeddings to K-Means Clusterer-------------------------------
NumClusters=3
KClusterer= KMeansClusterer(NumClusters,distance=nltk.cluster.util.euclidean_distance, repeats=200)
assigned_clusters = KClusterer.cluster(X, assign_clusters=True)
#print(assigned_clusters)

print(" Words and their assigned cluster numbers \n" + "="*50)
Words = list(model.wv.vocab)
for i, Word in enumerate(Words):
    print (Word + ":" + str(assigned_clusters[i]))

Pca = PCA(n_components=2)
Y= Pca.fit_transform(X)
pyplot.scatter(Y[:, 0], Y[:, 1], c=assigned_clusters, s=290, alpha=.5)

for j in range(len(Words)):
    pyplot.annotate(assigned_clusters[j], xy=(Y[j][0], Y[j][1]), xytext=(0, 0), textcoords='offset points')
pyplot.show()
