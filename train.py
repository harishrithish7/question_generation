import cPickle as pickle
import operator
import numpy as np
from model import TrainingModel
import random
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
from word2vec_preprocessing import word_vector_len
from gensim.models import KeyedVectors
from keras.models import load_model
import argparse
import re
import keras


epochs = 50
batch_size = 64
bin_size = 50
num_bins = 7
val_ratio = 0.1
train_ratio = 1-val_ratio

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

def DataGen(sentence, qn_output, qn_input, get_word_vector):
    indices = np.random.permutation(len(sentence))
    sentence, qn_output, qn_input = sentence[indices], qn_output[indices], qn_input[indices]

    sentence_bins = [[] for _ in xrange(num_bins)]
    question_output_bins = [[] for _ in xrange(num_bins)]
    question_input_bins = [[] for _ in xrange(num_bins)]


    def put(sent, qout, qin, idx):
        sentence_bins[idx].append(sent)
        question_output_bins[idx].append(qout)
        question_input_bins[idx].append(qin)

    idx = 0
    while True:
        if idx == len(sentence):
            idx = 0
            indices = np.random.permutation(len(sentence))
            sentence, qn_output, qn_input = sentence[indices], qn_output[indices], qn_input[indices]

        bin_idx = min(num_bins-1, len(sentence[idx]) // bin_size)
        put(sentence[idx], qn_output[idx], qn_input[idx], bin_idx)

        if len(sentence_bins[bin_idx]) == batch_size:
            embedded_sentence = [map(get_word_vector, sent) for sent in sentence_bins[bin_idx]]
            padded_sentence = pad_sequences(embedded_sentence, maxlen=len(max(embedded_sentence, key=len)), padding='post')
            
            embedded_question_input = [map(get_word_vector, question) for question in question_input_bins[bin_idx]]
            padded_question_input = pad_sequences(embedded_question_input, maxlen=len(max(embedded_question_input, key=len)), padding='post')
            
            padded_question_output = pad_sequences(question_output_bins[bin_idx], maxlen=len(max(question_output_bins[bin_idx], key=len)), padding='post')
            one_hot_question_output = to_categorical(padded_question_output, num_classes=word_vector_len)

            sentence_bins[bin_idx] = []
            question_output_bins[bin_idx] = []
            question_input_bins[bin_idx] = []
            
            yield [padded_sentence, padded_question_input], one_hot_question_output

        idx += 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--word2vec_path', type=str,
                        default='data/glove.6B.100d.trimmed.vec',
                        help='Word2Vec vectors file path')
    parser.add_argument('--data', type=str, default='data/preprocessed_data.pkl',
                        help='Desired path to output pickle')
    parser.add_argument('--path_to_load_weights', type=str, default=None,
                        help='Path to pre-trained weights')
    args = parser.parse_args()

    print "Loading training data"
    with open(args.data) as f:
        data = pickle.load(f)
    print "Done"

    get_word_vector = word2vec(args.word2vec_path)

    sentence, context, qn_output, qn_input, answer = operator.itemgetter("sentence", "context", "qn_output", "qn_input", "answer")(data)

    indices = np.random.permutation(len(sentence))
    sentence, context, qn_output, qn_input = sentence[indices], context[indices], qn_output[indices], qn_input[indices]

    num_train = int(len(sentence)*train_ratio)
    num_val = len(sentence)-num_train

    print("Creating Model")
    model = TrainingModel()
    print("Done!")

    if args.path_to_load_weights:
        print("Loading weights")
        model.load_weights(args.path_to_load_weights)
        print("Done!")

    train_data_generator = DataGen(sentence=sentence[:num_train], qn_output=qn_output[:num_train], qn_input=qn_input[:num_train], get_word_vector=get_word_vector)
    val_data_generator = DataGen(sentence=sentence[num_train:], qn_output=qn_output[num_train:], qn_input=qn_input[num_train:], get_word_vector=get_word_vector)

    checkpoint_best = ModelCheckpoint('pre-trained/best.hdf5', monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True)
    checkpoint_current = ModelCheckpoint('pre-trained/current.hdf5', monitor='val_loss', verbose=1, save_weights_only=True)
    model.fit_generator(train_data_generator, steps_per_epoch=num_train // batch_size, epochs=50, verbose=1, 
                        validation_data=val_data_generator, validation_steps=num_val // batch_size, callbacks=[checkpoint_best, checkpoint_current])
    

    """print("Creating Model")
    tr_model = TrainingModel()
    print("Done!")
    print("Loading weights")
    tr_model.load_weights("pre-trained/current.hdf5")
    print("Done!")

    tr_model.fit_generator(train_data_generator, steps_per_epoch=num_train // batch_size, epochs=4, verbose=1, 
                        validation_data=val_data_generator, validation_steps=num_val // batch_size)

    for layer in model.layers:
        if re.search('(input)|(concatenate)',layer.name):
            continue
        print layer.name
        print layer.get_weights()
        print np.array(layer.get_weights()).shape


    print("Creating Model")
    tr_model = TrainingModel()
    print("Done!")
    if args.path_to_load_weights:
        print("Loading weights")
        tr_model.load_weights(args.path_to_load_weights)
        print("Done!")

    
    for layer in tr_model.layers:
        if re.search('(input)|(concatenate)',layer.name):
            continue
        print layer.name
        print layer.get_weights()
        print layer.get_weights().shape

    
    tr_model.fit_generator(train_data_generator, steps_per_epoch=num_train // batch_size, epochs=7, verbose=1, 
                        validation_data=val_data_generator, validation_steps=num_val // batch_size, callbacks=[checkpoint_best, checkpoint_current])"""



