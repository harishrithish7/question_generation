from keras.models import Model
from keras.layers import Input, LSTM, Dense
import cPickle as pickle
import operator
from keras.utils.vis_utils import plot_model
from word2vec_preprocessing import embedding_dimension, word_vector_len

hidden_dim = 64
def TrainingModel(hidden_dim = hidden_dim):
	encoder_inputs = Input(shape=(None, embedding_dimension))
	encoder = LSTM(hidden_dim, return_state=True, name="encoder_lstm")
	encoder_outputs, state_h, state_c = encoder(encoder_inputs)

	encoder_states = [state_h, state_c]

	decoder_inputs = Input(shape=(None, embedding_dimension))
	decoder_lstm = LSTM(hidden_dim, return_sequences=True, return_state=True, name="decoder_lstm")
	decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
	decoder_dense = Dense(word_vector_len, activation="softmax", name="dense")
	decoder_outputs = decoder_dense(decoder_outputs)

	model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
	model.compile(optimizer="rmsprop", loss="categorical_crossentropy")


	plot_model(model, to_file='training_model.png', show_shapes=True)
	return model

def PredictionEncoderModel():
	encoder_inputs = Input(shape=(None, embedding_dimension))
	encoder = LSTM(hidden_dim, return_state=True, name="encoder_lstm")
	encoder_outputs, state_h, state_c = encoder(encoder_inputs)

	encoder_states = [state_h, state_c]

	model = Model(encoder_inputs, encoder_states)
	plot_model(model, to_file='prediction_encoder_model.png', show_shapes=True)
	return model

def PredictionDecoderModel():
	decoder_inputs = Input(shape=(None, embedding_dimension))

	decoder_state_input_h = Input(shape=(hidden_dim,))
	decoder_state_input_c = Input(shape=(hidden_dim,))
	decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

	decoder_lstm = LSTM(hidden_dim, return_sequences=True, return_state=True, name="decoder_lstm")
	decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
	decoder_states = [state_h, state_c]

	decoder_dense = Dense(word_vector_len, activation="softmax", name="dense")
	decoder_outputs = decoder_dense(decoder_outputs)
	model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

	plot_model(model, to_file='prediction_decoder_model.png', show_shapes=True)
	return model