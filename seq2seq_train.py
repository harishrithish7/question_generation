import cPickle as pickle
import operator
import numpy as np
import random
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
from word2vec_preprocessing import word_vector_len, embedding_dimension
from gensim.models import KeyedVectors
import argparse
import seq2seq
from seq2seq.models import Seq2Seq

epochs = 5
batch_size = 32
hidden_dim = 64
bin_size = 50
num_bins = 7

def word2vec(word2vec_path):
	print('Reading word2vec data... ')
	model = KeyedVectors.load_word2vec_format(word2vec_path)
	print('Done')

	def get_word_vector(word):
		try:
			return model[word.lower()]
		except KeyError:
			return model["<unk>"]
	return get_word_vector

def DataGen(context, qn_output, qn_input, get_word_vector):
	indices = np.random.permutation(len(context))
	context, qn_output, qn_input = context[indices], qn_output[indices], qn_input[indices]

	context_bins = [[] for _ in xrange(num_bins)]
	question_output_bins = [[] for _ in xrange(num_bins)]
	question_input_bins = [[] for _ in xrange(num_bins)]


	def put(c, qout, qin, idx):
		context_bins[idx].append(c)
		question_output_bins[idx].append(qout)
		question_input_bins[idx].append(qin)

	idx = 0
	while True:
		if idx == len(context):
			idx = 0
			indices = np.random.permutation(len(context))
			context, qn_output, qn_input = context[indices], qn_output[indices], qn_input[indices]

		bin_idx = min(num_bins-1, len(context[idx]) // bin_size)
		put(context[idx], qn_output[idx], qn_input[idx], bin_idx)

		if len(context_bins[bin_idx]) == batch_size:
			embedded_context = [map(get_word_vector, cxt) for cxt in context_bins[bin_idx]]
			padded_context = pad_sequences(embedded_context, maxlen=len(max(embedded_context, key=len)), padding='post')
			
			#embedded_question_input = [map(get_word_vector, question) for question in question_input_bins[bin_idx]]
			embedded_question_input = [map(get_word_vector, question[1:]) for question in question_input_bins[bin_idx]]
			padded_question_input = pad_sequences(embedded_question_input, maxlen=len(max(embedded_question_input, key=len)), padding='post')
			
			padded_question_output = pad_sequences(question_output_bins[bin_idx], maxlen=len(max(question_output_bins[bin_idx], key=len)), padding='post')
			one_hot_question_output = to_categorical(padded_question_output, num_classes=word_vector_len)

			context_bins[bin_idx] = []
			question_output_bins[bin_idx] = []
			question_input_bins[bin_idx] = []
			
			#yield [padded_context, padded_question_input], one_hot_question_output
			yield padded_context, padded_question_input

		idx += 1

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--word2vec_path', type=str,
						default='data/glove.6B.100d.trimmed.vec',
						help='Word2Vec vectors file path')
	parser.add_argument('--data', type=str, default='data/preprocessed_data.pkl',
                        help='Desired path to output pickle')
	args = parser.parse_args()

	print "Loading training data"
	with open(args.data) as f:
		data = pickle.load(f)
	print "Done"

	get_word_vector = word2vec(args.word2vec_path)

	context, qn_output, qn_input, answer = operator.itemgetter("context", "qn_output", "qn_input", "answer")(data)
	num_train = len(context)
	context, qn_output, qn_input, answer = np.array(context), np.array(qn_output), np.array(qn_input), np.array(answer)

	model = Seq2Seq(input_dim=embedding_dimension, batch_size=batch_size, hidden_dim=hidden_dim, 
							output_length=word_vector_len, output_dim=embedding_dimension, broadcast_state=True, peek=True)
	model.compile(loss='mse', optimizer='rmsprop')
	data_generator = DataGen(context=context, qn_output=qn_output, qn_input=qn_input, get_word_vector=get_word_vector)

	checkpoint = ModelCheckpoint('model/model.{epoch:02d}.hdf5')
	model.fit_generator(data_generator, steps_per_epoch=num_train // batch_size, epochs=epochs, verbose=1, callbacks=[checkpoint])



