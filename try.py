import json
from keras.preprocessing.text import Tokenizer,one_hot, text_to_word_sequence
from unidecode import unidecode
import cPickle as pickle
import numpy as np

"""with open("data/train_parsed_record.json") as f:
    samples = json.load(f)

def parse_sample(context, question, topic, answer_start, **kwargs):
    context = text_to_word_sequence(context)
    context = [unidecode(token) for token in context]
    context = ' '.join(context)
    print context

samples = [parse_sample(**sample) for sample in samples[:1]]"""
"""with open("data/valid_parsed_sep.json") as f:
    sep = json.load(f)

print sep.keys()     

for key in sep.keys():
    print sep[key][0]"""


"""with open("data/train_parsed_sep.json") as f:
    samples = json.load(f)

length = []
for context in samples["context"]:
    length.append(len(context.split(' ')))
print sum(length)/len(length)
print len(length)
print max(length)"""

""""with open("data/parsed_data.pkl") as f:
    data = pickle.load(f)
print "Done"

conlen = []
for context in data["context"]:
    conlen.append(len(context))

qnlen = []
for question in data["question"]:
    qnlen.append(len(question))

length = {
    "conlen": conlen,
    "qnlen": qnlen
}

with open("data/length.pkl","wb") as f:
    pickle.dump(length, f, protocol=pickle.HIGHEST_PROTOCOL)"""

"""with open("data/length.pkl") as f:
    data = pickle.load(f)

conlen, qnlen = data["conlen"], data["qnlen"]

import matplotlib.pyplot as plt
bins = [0,50,100,150,200,250,300]
n, bins, patches = plt.hist(conlen, bins)
print n
print bins
print patches
plt.show()"""

"""print "Loading data"
with open("data/preprocessed_data.pkl") as f:
    data = pickle.load(f)
print "Done" 

reclen = 800
context =  np.asarray(data["context"][:reclen])
qn_output = np.asarray(data["qn_output"][:reclen])
answer = np.asarray(data["answer"][:reclen])
qn_input = np.asarray(data["qn_input"][:reclen])

data = {
    "context": context,
    "qn_output": qn_output,
    "qn_input": qn_input,
    "answer": answer
}

with open("data/preprocessed_data_trimmed.pkl", "wb") as f:
    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)"""

from gensim.models import KeyedVectors
import argparse


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
    args = parser.parse_args()
    word_vector, get_word_vector = word2vec(args.word2vec_path)

    with open("data/preprocessed_data_trimmed.pkl") as f:
        data = pickle.load(f)
    print "Done" 

    print [map(get_word_vector, item) for item in data["qn_input"][:2]]


from keras.models import Model
from keras.layers import Input
from keras.layers import LSTM
from keras.layers import Dense
from keras.utils.vis_utils import plot_model
# configure
num_encoder_tokens = 71
num_decoder_tokens = 93
latent_dim = 256
# Define an input sequence and process it.
encoder_inputs = Input(shape=(None, num_encoder_tokens))
encoder = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)
# We discard `encoder_outputs` and only keep the states.
encoder_states = [state_h, state_c]
# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = Input(shape=(None, num_decoder_tokens))
# We set up our decoder to return full output sequences,
# and to return internal states as well. We don't use the
# return states in the training model, but we will use them in inference.
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)
# Define the model that will turn
# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
# plot the model
plot_model(model, to_file='model.png', show_shapes=True)
# define encoder inference model
encoder_model = Model(encoder_inputs, encoder_states)
# define decoder inference model
decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)
# summarize model
plot_model(encoder_model, to_file='encoder_model.png', show_shapes=True)
plot_model(decoder_model, to_file='decoder_model.png', show_shapes=True)


#Define an input sequence and process it.
encoder_inputs = Input(shape=(None, num_encoder_tokens))
encoder = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)
# We discard `encoder_outputs` and only keep the states.
encoder_states = [state_h, state_c]
encoder_model = Model(encoder_inputs, encoder_states)

# define decoder inference model
decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = Input(shape=(None, num_decoder_tokens))

decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)

decoder_states = [state_h, state_c]
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)




