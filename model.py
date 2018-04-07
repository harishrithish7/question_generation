from keras.models import Model
from keras.layers import Input, LSTM, Dense, Bidirectional, Concatenate, Softmax
import cPickle as pickle
from keras.utils.vis_utils import plot_model
from word2vec_preprocessing import embedding_dimension, word_vector_len
import numpy as np
import tensorflow as K
from keras import optimizers

#from keras import backend as K
from keras.engine.topology import Layer

hidden_dim = 600

"""class EncodeContext(Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(EncodeContext, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W_t = self.add_weight(name='W_t', 
                                      shape=(1, 1, input_shape[2]),
                                      initializer='glorot_uniform',
                                      trainable=True)
        super(EncodeContext, self).build(input_shape) 

    def call(self, x):
        W_t_broadcast = K.tile(self.W_t, [K.shape(x)[0],K.shape(x)[1],1])

        tanh_input = K.multiply(W_t_broadcast, x)
        tanh_output = K.tanh(tanh_input)

        softmax_inputs = K.matmul(tanh_output, self.W_s)
        softmax = Softmax(axis=-1)
        softmax_outputs = softmax(softmax_inputs)
        return softmax_outputs

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.output_dim)"""

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

        W_b_broadcast = K.tile(self.W_b, [K.shape(h)[0],1,1])

        tmp1 = K.matmul(h,  W_b_broadcast)
        b_trans = K.transpose(b, perm=[0,2,1])
        tmp2 = K.matmul(tmp1, b_trans)

        softmax = Softmax(axis=-1)
        softmax_outputs = softmax(tmp2)

        softmax_outputs_tiled = K.tile(softmax_outputs, [1,1,b_shape[2]])
        shape = K.shape(softmax_outputs)
        softmax_outputs_broadcast = K.reshape(softmax_outputs_tiled, [shape[0],shape[1],shape[2],b_shape[2]])

        shape = K.shape(b)
        b_reshaped = K.reshape(b, [shape[0],1,shape[1],shape[2]])
        qn_length = K.shape(h)[1]
        b_broadcast = K.tile(b_reshaped, [1,qn_length,1,1])

        cxt_vector_unsummed = K.multiply(softmax_outputs_broadcast, b_broadcast)
        cxt_vector_unshaped = K.reduce_sum(cxt_vector_unsummed, axis=2)
        shape = K.shape(cxt_vector_unshaped)
        cxt_vector = K.reshape(cxt_vector_unshaped, [shape[0],shape[1],shape[2]])

        return cxt_vector

    def compute_output_shape(self, input_shape):
        b_shape, h_shape = input_shape
        return (h_shape[0], h_shape[1], b_shape[2])

def TrainingModel():
    encoder_inputs = Input(shape=(None, embedding_dimension))
    encoder = Bidirectional(LSTM(hidden_dim, return_state=True, return_sequences=True, name="encoder_lstm"), merge_mode='concat', name="bidirectional_lstm")
    encoder_outputs, forward_h, forward_c, backward_h, backward_c = encoder(encoder_inputs)

    state_h = Concatenate()([forward_h, backward_h])
    state_c = Concatenate()([forward_c, backward_c])
    encoder_states = [state_h, state_c]


    decoder_inputs = Input(shape=(None, embedding_dimension))
    decoder_lstm = LSTM(2*hidden_dim, return_sequences=True, return_state=True, name="decoder_lstm")
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)

    attention = Attention(name="attention")
    cxt_vector = attention([encoder_outputs, decoder_outputs])

    encode_context_input = Concatenate()([decoder_outputs, cxt_vector])

    #encode_context = EncodeContext(output_dim=word_vector_len, name="encode_context")
    #encoded_output = encode_context(encode_context_input)

    """model = Model([encoder_inputs, decoder_inputs], encoded_output)"""

    decoder_dense = Dense(word_vector_len, activation="softmax", name="dense")
    outputs = decoder_dense(encode_context_input)
    model = Model([encoder_inputs, decoder_inputs], outputs)

    sgd = optimizer.SGD(lr=1.0)

    model.compile(optimizer="rmsprop", loss="categorical_crossentropy")
    plot_model(model, to_file='training_model.png', show_shapes=True)
    return model

def PredictionEncoderModel():
    encoder_inputs = Input(shape=(None, embedding_dimension))
    encoder = Bidirectional(LSTM(hidden_dim, return_state=True, return_sequences=True, name="encoder_lstm"), merge_mode='concat', name="bidirectional_lstm")
    encoder_outputs, forward_h, forward_c, backward_h, backward_c = encoder(encoder_inputs)

    state_h = Concatenate()([forward_h, backward_h])
    state_c = Concatenate()([forward_c, backward_c])
    encoder_states = [state_h, state_c]

    model = Model(encoder_inputs, [encoder_outputs] + encoder_states)
    plot_model(model, to_file='prediction_encoder_model.png', show_shapes=True)
    return model

def PredictionDecoderModel():
    decoder_inputs = Input(shape=(None, embedding_dimension))

    decoder_state_input_h = Input(shape=(2*hidden_dim,))
    decoder_state_input_c = Input(shape=(2*hidden_dim,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

    decoder_lstm = LSTM(2*hidden_dim, return_sequences=True, return_state=True, name="decoder_lstm")
    decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]

    encoder_outputs = Input(shape=(None, 2*hidden_dim))

    attention = Attention(name="attention")
    cxt_vector = attention([encoder_outputs, decoder_outputs])

    encode_context_input = Concatenate()([decoder_outputs, cxt_vector])

    #encode_context = EncodeContext(word_vector_len, name="encode_context")
    #encoded_output = encode_context(encode_context_input)

    decoder_dense = Dense(word_vector_len, activation="softmax", name="dense")
    outputs = decoder_dense(encode_context_input)

    model = Model([decoder_inputs] + decoder_states_inputs + [encoder_outputs], [outputs] + decoder_states)
    plot_model(model, to_file='training_model.png', show_shapes=True)
    return model