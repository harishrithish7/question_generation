from gensim.scripts.glove2word2vec import glove2word2vec
from random import random

word_vector_len = 20000 + 3
embedding_dimension = 100
with open("data/glove.6B.100d.txt") as f:
	with open("data/glove.6B.100d.trimmed.txt", "w") as f1:
		f1.write("<start> ")
		for i in xrange(embedding_dimension):
			f1.write( str("%.6f" % (random()*2.0-1.0)) )
			if i == embedding_dimension-1:
				f1.write("\n")
			else:
				f1.write(" ")

		f1.write("<end> ")
		for i in xrange(embedding_dimension):
			f1.write( str("%.6f" % (random()*2.0-1.0)) )
			if i == embedding_dimension-1:
				f1.write("\n")
			else:
				f1.write(" ")

		f1.write("<unk> ")
		for i in xrange(embedding_dimension):
			f1.write( str("%.6f" % (random()*2.0-1.0)) )
			if i == embedding_dimension-1:
				f1.write("\n")
			else:
				f1.write(" ")

		i = 3
		for line in f:
			f1.write(line)
			i += 1
			if i == word_vector_len:
				break


glove2word2vec("data/glove.6B.100d.trimmed.txt", "data/glove.6B.100d.trimmed.vec")
