from nltk.corpus import brown
from nltk import word_tokenize
import re
from gensim import models,corpora
from nltk.corpus import stopwords

Num_topics= 10
StopWords= stopwords.words('english')

def clean_text(text):
    tokenized_text=word_tokenize(text.lower())
    cleaned_text=[t for t in tokenized_text if t not in StopWords and re.match('[a-zA-Z\-][a-zA-Z\-]{2,}', t)]
    return cleaned_text

data= []
for fileid in brown.fileids():
    document= ' '.join(brown.words(fileid))
    data.append(document)
No_Documents= len(data)

#Preprocessing the brown corpus
tokenized_data=[]
for text in data:
    tokenized_data.append(clean_text(text))
#print(tokenized_data[:2])

#Creating a Dictionary of the words in brown corpus
dictionary= corpora.Dictionary(tokenized_data)

#Creating a bag fo words with the corpus
Bow= [dictionary.doc2bow(text)for text in tokenized_data]

#Printing the 20th document (WordID, WordCount) output
#print(Bow[20])

#LDA model building
ldaModel=models.LdaModel(corpus=Bow,num_topics=Num_topics,id2word=dictionary)
print("LDA Model")
for idx in range(Num_topics):
    print("Topic #%s:" %idx, ldaModel.print_topic(idx,10))
print("=" *200)

#Training Model on a Test Data
text=" the life of ghanaians is improving day by day"
bow=dictionary.doc2bow(clean_text(text))
print(ldaModel[bow])