import gensim
from sklearn.decomposition import PCA
from matplotlib import pyplot
from gensim.models import KeyedVectors

model = KeyedVectors.load("MyModel")
print(model)
print(model.most_similar("akufo"))
print(model.most_similar("military"))
#print(model.most_similar("protest"))
#print(model["ghana"])


print(model.wv.syn0.shape)


X = model[model.wv.vocab]
pca = PCA(n_components=2)
result = pca.fit_transform(X)
# create a scatter plot of the projection
pyplot.scatter(result[:, 0], result[:, 1])
words = list(model.wv.vocab)
for i, word in enumerate(words):
	pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
pyplot.title("Word Embeddings of US Military Base Data")
pyplot.show()

