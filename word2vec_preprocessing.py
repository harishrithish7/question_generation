# Assumes encoder_tokens > decoder_tokens

from gensim.scripts.glove2word2vec import glove2word2vec
from random import random

embedding_dimension = 300
encoder_tokens = 45000+3
decoder_tokens = 28000+3
num_words = 840 # in billions

def produce_vector():
	return [str("%.6f" % (random()*2.0-1.0)) for _ in xrange(embedding_dimension)].join(' ')[:-1]

start = produce_vector()
end = produce_vector()
unk = produce_vector()

with open("data/glove.{}B.{}.encoder.txt".format(num_words, embedding_dimension)) as f:
	with open("data/glove.{}B.{}.trimmed.decoder.txt".format(num_words, embedding_dimension), "w") as f1:
		with open("data/glove.{}B.{}.trimmed.encoder.txt".format(num_words, embedding_dimension), "w") as f2:

			f1.write("<start> ")
			f1.write(start)
			f1.write("\n")

			f1.write("<end> ")
			f1.write(end)
			f1.write("\n")

			f1.write("<unk> ")
			f1.write(unk)
			f1.write("\n")

			i = 3
			for line in f:
				f1.write(line)
				i += 1
				if i == encoder_tokens:
					break
glove2word2vec("data/glove.{}B.{}d.trimmed.encoder.txt".format(num_words, embedding_dimension), "data/glove.{}B.{}d.trimmed.encoder.vec".format(num_words, embedding_dimension))


with open("data/glove.{}B.{}.decoder.txt".format(num_words, embedding_dimension)) as f:
	with open("data/glove.{}B.{}.trimmed.decoder.txt".format(num_words, embedding_dimension), "w") as f1:
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
			if i == decoder_tokens:
				break
glove2word2vec("data/glove.{}B.{}d.trimmed.decoder.txt".format(num_words, embedding_dimension), "data/glove.{}B.{}d.trimmed.decoder.vec".format(num_words, embedding_dimension))
