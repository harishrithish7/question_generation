from keras.models import Model
from keras.layers import Input, LSTM, Dense
import cPickle as pickle
import operator
from keras.utils.vis_utils import plot_model
from word2vec_preprocessing import embedding_dimension, word_vector_len

def Seq2SeqModel(hidden_dim = 64):
	encoder_inputs = Input(shape=(None, embedding_dimension))
	encoder = LSTM(hidden_dim, return_state=True)
	encoder_outputs, state_h, state_c = encoder(encoder_inputs)

	encoder_states = [state_h, state_c]

	decoder_inputs = Input(shape=(None, embedding_dimension))
	decoder_lstm = LSTM(hidden_dim, return_sequences=True, return_state=True)
	decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
	decoder_dense = Dense(word_vector_len, activation="softmax")
	decoder_outputs = decoder_dense(decoder_outputs)

	model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
	model.compile(optimizer="rmsprop", loss="categorical_crossentropy")
	plot_model(model, to_file='model.png', show_shapes=True)
	return model

	