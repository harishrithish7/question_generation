from keras.models import Model, load_model, Sequential
from gensim.models import KeyedVectors
import argparse
import cPickle as pickle
import operator
from model import PredictionEncoderModel, PredictionDecoderModel, TrainingModel
from word2vec_preprocessing import embedding_dimension
import numpy as np
import re

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
	parser.add_argument('--path_to_load_weights', type=str, default="pre-trained/best.hdf5",
						help="Path to pre-trained model")
	args = parser.parse_args()

	print "Loading training data"
	with open(args.data) as f:
		data = pickle.load(f)
	print "Done" 

	word_vector, get_word_vector = word2vec(args.word2vec_path)

	index_word_map = dict((idx,word) for idx, word in enumerate(sorted(word_vector.vocab.keys())) )

	sentence, context, qn_output, qn_input, answer = operator.itemgetter("sentence", "context", "qn_output", "qn_input", "answer")(data)
	

	trained_model = TrainingModel()
	trained_model.load_weights(args.path_to_load_weights)

	encoder_model = PredictionEncoderModel()
	for layer in encoder_model.layers:
		if re.search('(input)|(concatenate)',layer.name):
			continue
		encoder_model.get_layer(layer.name).set_weights(trained_model.get_layer(layer.name).get_weights())

	decoder_model = PredictionDecoderModel()
	for layer in decoder_model.layers:
		if re.search('(input)|(concatenate)',layer.name):
			continue
		decoder_model.get_layer(layer.name).set_weights(trained_model.get_layer(layer.name).get_weights())

	max_decoder_seq_length = 200
	start = 0
	end = 100
	for sent, qn in zip(sentence[start:end:5], qn_input[start:end:5]):
		print qn
		print sent
		embedded_sent = [get_word_vector(token) for token in sent]
		encoder_input = np.array(embedded_sent).reshape((1,len(embedded_sent),embedding_dimension))
		encoder_outputs, state_h, state_c = encoder_model.predict(encoder_input)

		states_value = [state_h, state_c]

		decoder_token_vector = get_word_vector("<start>")
		decoder_input = np.array(decoder_token_vector).reshape((1,1,embedding_dimension))


		stop_condition = False
		decoded_sentence = ''

		qn_idx = 1
		while not stop_condition:
			output_token, h, c = decoder_model.predict([decoder_input] + states_value + [encoder_outputs])

			token_index = np.argmax(output_token)
			token = index_word_map[token_index]
			decoded_sentence += token + " "

			if (token == '<end>' or len(decoded_sentence) > max_decoder_seq_length):
				stop_condition = True

			decoder_token_vector = get_word_vector(token)
			decoder_input = np.array(decoder_token_vector).reshape((1,1,embedding_dimension))
			
			"""token = qn[min(qn_idx, len(qn)-1)]
			print token
			decoder_token_vector = get_word_vector(token)
			decoder_input = np.array(decoder_token_vector).reshape((1,1,embedding_dimension))
			qn_idx += 1"""

			states_value = [h, c]

		print decoded_sentence



