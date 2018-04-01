from keras.models import Model
from model import PredictionEncoderModel, PredictionDecoderModel
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Bidirectional, Concatenate, Softmax
import cPickle as pickle
import operator
from keras.utils.vis_utils import plot_model
from word2vec_preprocessing import embedding_dimension, word_vector_len
import numpy as np
import tensorflow as tf


#from keras import backend as K
from keras.engine.topology import Layer
import tensorflow as K

class Attention(Layer):

    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        b_shape, h_shape = input_shape
        self.W_b = self.add_weight(name='W_b', 
                                      shape=(1, b_shape[2], b_shape[2]),
                                      initializer='glorot_uniform',
                                      trainable=True)
        super(Attention, self).build(input_shape) 

    def call(self, x):
        b, h = x
        b_shape = K.shape(b)

        tmp1 = K.matmul(h, self.W_b)
        b_trans = K.transpose(b, perm=[0,2,1])
        tmp2 = K.matmul(tmp1, b_trans)

        softmax = Softmax(axis=-1)
        softmax_outputs = softmax(tmp2)

        softmax_outputs_tiled = K.tile(softmax_outputs, [1,1,b_shape[2]])
        softmax_outputs_broadcast = K.reshape(softmax_outputs_tiled, K.shape(softmax_outputs_tiled)+[b_shape[2]])

        shape = K.shape(b)
        b_reshaped = K.reshape(b, [shape[0],1,shape[1],shape[2]])
        b_broadcast = K.tile(b_reshaped, [1,b_shape[2],1,1])

        cxt_vector_unsummed = K.multiply(softmax_outputs_broadcast, b_broadcast)
        cxt_vector_unshaped = K.reduce_sum(cxt_vector_unsummed, axis=2)
        shape = K.shape(cxt_vector_unshaped)
        cxt_vector = K.reshape(cxt_vector_unshaped, [shape[0],shape[1],shape[2]])

        return cxt_vector

    def compute_output_shape(self, input_shape):
        b_shape, h_shape = input_shape
        return (h_shape[0], h_shape[1], b_shape[2])

class AdvancedSoftmax(Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(AdvancedSoftmax, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W_t = self.add_weight(name='W_t', 
                                      shape=(1, 1, input_shape[2]),
                                      initializer='glorot_uniform',
                                      trainable=True)
        self.W_s = self.add_weight(name='W_s', 
                                      shape=(1, input_shape[2], self.output_dim),
                                      initializer='glorot_uniform',
                                      trainable=True)
        super(AdvancedSoftmax, self).build(input_shape) 

    def call(self, x):
        W_t_broadcast = K.tile(self.W_t, [K.shape(x)[0],K.shape(x)[1],1])

        tanh_input = K.multiply(W_t_broadcast, x)
        tanh_output = K.tanh(tanh_input)

        softmax_inputs = K.matmul(tanh_output, self.W_s)
        softmax = Softmax(axis=-1)
        softmax_outputs = softmax(softmax_inputs)

        return softmax_outputs

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.output_dim)



hidden_dim = 256

encoder_inputs = Input(shape=(75, embedding_dimension))
encoder = Bidirectional(LSTM(hidden_dim, return_state=True, return_sequences=True, name="encoder_lstm"), merge_mode='concat')
encoder_outputs, forward_h, forward_c, backward_h, backward_c = encoder(encoder_inputs)

state_h = Concatenate()([forward_h, backward_h])
state_c = Concatenate()([forward_c, backward_c])
encoder_states = [state_h, state_c]


decoder_inputs = Input(shape=(10, embedding_dimension))
decoder_lstm = LSTM(2*hidden_dim, return_sequences=True, return_state=True, name="decoder_lstm")
decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)

attention = Attention()
cxt_vector = attention([encoder_outputs, decoder_outputs])

adv_softmax_input = Concatenate()([decoder_outputs, cxt_vector])

adv_softmax = AdvancedSoftmax(word_vector_len)
output = adv_softmax(adv_softmax_input)

model = Model([encoder_inputs, decoder_inputs], output)
model.compile(optimizer="rmsprop", loss="categorical_crossentropy")

plot_model(model, to_file='training_model.png', show_shapes=True)
