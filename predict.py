from keras.models import Model, load_model
from gensim.models import KeyedVectors
import argparse
import cPickle as pickle
import operator
from model import PredictionEncoderModel, PredictionDecoderModel
from word2vec_preprocessing import embedding_dimension
import numpy as np

def word2vec(word2vec_path):
	print('Reading word2vec data... ')
	model = KeyedVectors.load_word2vec_format(word2vec_path)
	print('Done')

	def get_word_vector(word):
		try:
			return model[word.lower()]
		except KeyError:
			return model["<unk>"]
	return model, get_word_vector

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--word2vec_path', type=str,
						default='data/glove.6B.100d.trimmed.vec',
						help='Word2Vec vectors file path')
	parser.add_argument('--data', type=str, default='data/preprocessed_data_trimmed.pkl',
						help='Desired path to output pickle')
	args = parser.parse_args()

	print "Loading training data"
	with open(args.data) as f:
		data = pickle.load(f)
	print "Done"

	word_vector, get_word_vector = word2vec(args.word2vec_path)

	index_word_map = dict((idx,word) for idx, word in enumerate(sorted(word_vector.vocab.keys())) )

	context, qn_output, qn_input, answer = operator.itemgetter("context", "qn_output", "qn_input", "answer")(data)

	train_model = load_model("model/model.05.hdf5")
	encoder_model = PredictionEncoderModel()
	encoder_model.get_layer("encoder_lstm").set_weights(train_model.get_layer("encoder_lstm").get_weights())

	decoder_model = PredictionDecoderModel()
	decoder_model.get_layer("decoder_lstm").set_weights(train_model.get_layer("decoder_lstm").get_weights())
	decoder_model.get_layer("dense").set_weights(train_model.get_layer("dense").get_weights())

	max_decoder_seq_length = 200
	start = 0
	end = 1
	for cxt, qn in zip(context[start:end], qn_input[start:end]):
		print cxt
		embedded_context = [get_word_vector(token) for token in cxt]
		encoder_input = np.array(embedded_context).reshape((1,len(embedded_context),embedding_dimension))
		states_value = encoder_model.predict(encoder_input)

		decoder_token_vector = get_word_vector("<start>")
		decoder_input = np.array(decoder_token_vector).reshape((1,1,embedding_dimension))


		stop_condition = False
		decoded_sentence = ''

		qn_idx = 1
		while not stop_condition:
			output_token, h, c = decoder_model.predict([decoder_input] + states_value)

			token_index = np.argmax(output_token)
			token = index_word_map[token_index]
			decoded_sentence += token + " "

			if (token == '<unk>' or len(decoded_sentence) > max_decoder_seq_length):
				stop_condition = True

			#decoder_token_vector = get_word_vector(token)
			#decoder_input = np.array(decoder_token_vector).reshape((1,1,embedding_dimension))
			token = qn[min(qn_idx, len(qn)-1)]
			print token
			decoder_token_vector = get_word_vector(token)
			decoder_input = np.array(decoder_token_vector).reshape((1,1,embedding_dimension))
			qn_idx += 1

			states_value = [h, c]

		print decoded_sentence



