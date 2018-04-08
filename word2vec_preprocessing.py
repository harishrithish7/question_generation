# Assumes encoder_tokens > decoder_tokens

from gensim.scripts.glove2word2vec import glove2word2vec
from random import random

embedding_dimension = 300
encoder_tokens = 45000+3
decoder_tokens = 28000+3
corpus = 840 # in billions

def produce_vector():
	return ' '.join([str("%.6f" % (random()*2.0-1.0)) for _ in xrange(embedding_dimension)])[:-1]


if __name__ == '__main__':
	start = produce_vector()
	end = produce_vector()
	unk = produce_vector()

	tokens_added = []

	print("Writing encoder and decoder to .txt file")
	with open("data/glove.{}B.{}d.txt".format(corpus, embedding_dimension)) as f:
		with open("data/glove.{}B.{}d.decoder.txt".format(corpus, embedding_dimension), "w") as f1:
			with open("data/glove.{}B.{}d.encoder.txt".format(corpus, embedding_dimension), "w") as f2:
				for file in [f1,f2]:
					file.write("<start> ")
					file.write(start)
					file.write("\n")

					file.write("<end> ")
					file.write(end)
					file.write("\n")

					file.write("<unk> ")
					file.write(unk)
					file.write("\n")

				tokens = 3
				for line in f:
					if line.split(' ')[0].lower() in tokens_added:
						continue
					files = [f1,f2] if tokens < decoder_tokens else [f2]
					for file in files:
						file.write(line.lower())
					tokens += 1
					tokens_added.append(line.split(' ')[0].lower())
					if tokens == encoder_tokens:
						break
	print("Done!")

	print("Converting encoder vector from .txt to .vec")
	glove2word2vec("data/glove.{}B.{}d.encoder.txt".format(corpus, embedding_dimension), "data/glove.{}B.{}d.encoder.vec".format(corpus, embedding_dimension))
	print("Done!")

	print("Converting decoder vector from .txt to .vec")
	glove2word2vec("data/glove.{}B.{}d.decoder.txt".format(corpus, embedding_dimension), "data/glove.{}B.{}d.decoder.vec".format(corpus, embedding_dimension))
	print("Done!")