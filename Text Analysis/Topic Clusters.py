
from nltk import word_tokenize
import re
import pickle
import gensim
from gensim import models,corpora
from nltk.corpus import stopwords
import pyLDAvis
import pyLDAvis.gensim


Num_topics= 3
StopWords= stopwords.words('english')

Data= 'During inspection it was found that the POS1 ' \
      'strut is leaking and requires replacement Raise notification :' \
      ' 2 x fitters* Machine will be down*28.07.2018 11:06:18 UTC Damon Stacey (STACDA9) ' \
      'THIS JOB HAS BEEN RESCHEDULED 14.09.2018 03:24:15 UTC Scott Williams (WILLSO9) Advised by site that hub i' \
      's also leaking now Last time this corner was off a new strut was fitted but a second hand hub was all that was available at the time.' \
      ' I have added the material number for a complete as the hub and strut needs to be changed. Komatsu have been engaged ' \
      'for warranty on the strut. See attached email for photos from site. 16.10.2018 10:50:13 UTC Traven Hooper (HOOPT9)' \
      ' Job has been completed 16.10.2018 20:49:54 UTC Gavin Casten (CASTG91) Phone +6149805626 Work completed ahead of schedule'


def clean_text(text):
    tokenized_text=word_tokenize(text.lower())
    cleaned_text=[t for t in tokenized_text if t not in StopWords and re.match('[a-zA-Z\-][a-zA-Z\-]{2,}', t)]
    return cleaned_text

data= []
for word in Data.split(" "):
    data.append(word)
No_Documents= len(data)

#Preprocessing the brown corpus
tokenized_data=[]
for text in data:
    tokenized_data.append(clean_text(text))


#Creating a Dictionary of the words in brown corpus
dictionary= corpora.Dictionary(tokenized_data)

#----------------------------Creating a bag fo words with the corpus-------------------
Bow= [dictionary.doc2bow(text)for text in tokenized_data]
pickle.dump(Bow,open('Bow.pkl','wb'))
dictionary.save('dictionary.gensim')


#----------------------------------------LDA model building----------------------------
ldaModel=models.LdaModel(corpus=Bow,num_topics=Num_topics,id2word=dictionary)
ldaModel.save('ldaModel.gensim')
print("LDA Model")
for idx in range(Num_topics):
    print("Topic #%s:" %idx, ldaModel.print_topic(idx,20))
print("=" *200)

#----------------Assigning Topics to New data set---------------------------------------
text=" we completed the signing of the warranty document with komatsu"
bow=dictionary.doc2bow(clean_text(text))
print(ldaModel[bow])

#---------------------------Visualizing the Topic Models-----------------------------------

dictionary = gensim.corpora.Dictionary.load('dictionary.gensim')
Bow= pickle.load(open('Bow.pkl','rb'))
ldaModel=gensim.models.LdaModel.load('ldaModel.gensim')
ldaDisplay = pyLDAvis.gensim.prepare(topic_model=ldaModel, corpus=Bow, dictionary=dictionary)
pyLDAvis.display(ldaDisplay)
