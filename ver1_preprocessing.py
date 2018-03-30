import numpy as np
import json
import os
import argparse
import cPickle as pickle

from os import path
from gensim.scripts.glove2word2vec import glove2word2vec
from tqdm import tqdm
from unidecode import unidecode

# from utils import CoreNLP_path, get_glove_file_path
from gensim.models import KeyedVectors
from keras.preprocessing.sequence import pad_sequences

from keras.preprocessing.text import Tokenizer, text_to_word_sequence

from word2vec_preprocessing import word_vector_len


def word2vec(word2vec_path):
    print('Reading word2vec data... ')
    model = KeyedVectors.load_word2vec_format(word2vec_path)
    print('Done')

    def get_word_vector(word):
        try:
            return model[word]
        except KeyError:
            return model["<unk>"]
    return model, get_word_vector

def index2vec(tokenizer, get_word_vector):
    index_vector = {}
    print('Mapping index2vec ... ')
    for word, index in tokenizer.word_index.iteritems():
        index_vector[index] = get_word_vector(word)
    print('Done')
    return index_vector

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--word2vec_path', type=str,
                        default='data/glove.6B.100d.trimmed.vec',
                        help='Word2Vec vectors file path')
    parser.add_argument('--outfile', type=str, default='data/parsed_data.pkl',
                        help='Desired path to output pickle')
    parser.add_argument('--data', type=str, default="data/train_parsed_record.json",
                        help='Data json')
    args = parser.parse_args()

    if not args.outfile.endswith('.pkl'):
        args.outfile += '.pkl'

    print('Reading SQuAD data... ')
    with open(args.data) as fd:
        samples = json.load(fd)
    print('Done!')

    word_vector, get_word_vector = word2vec(args.word2vec_path)

    contexts = []
    questions = []
    answers = []

    # converts unicode to ascii
    print("Parsing samples")
    def parse_sample(context, question, answer, **kwargs):
        context = text_to_word_sequence(context)
        context = [unidecode(token) for token in context]
        context_vecs = [get_word_vector(token) for token in context]
        context_vecs.append(get_word_vector("<end>"))
        # context = ' '.join(context)
        contexts.append(context_vecs)

        question = text_to_word_sequence(question)
        question = [unidecode(token) for token in question]
        question_vecs = [get_word_vector(token) for token in question]
        #question = ' '.join(question)
        questions.append(question_vecs)

        answer = text_to_word_sequence(answer)
        answer = [unidecode(token) for token in answer]
        answer_vecs = [get_word_vector(token) for token in answer]
        #answer = ' '.join(answer)
        answers.append(answer_vecs)

        return {
                "context": context,
                "question": question,
                "answer": answer
            }

    samples = [parse_sample(**sample) for sample in tqdm(samples)]
    """ vocabulary = contexts + questions + answers
    tokenizer = Tokenizer(num_words=None)

    # gives an index to each word appearing in the text
    print("Fitting on text")
    tokenizer.fit_on_texts(vocabulary)
    print("Done")

    print("Converting text to sequences")
    context_index = tokenizer.texts_to_sequences(contexts)
    question_index = tokenizer.texts_to_sequences(questions)
    answer_index = tokenizer.texts_to_sequences(answers)
    print("Done")

    
    index_vector = index2vec(tokenizer, get_word_vector)

    data = {
        "word_vector": word_vector,
        "index_vector": index_vector,
        "context_index": context_index,
        "question_index": question_index,
        "answer_index": answer_index
    }"""

    data = {
        "context": contexts,
        "question": questions,
        "answer": answers
    }

    print('Writing to file {}... '.format(args.outfile))
    with open(args.outfile, 'wb') as fd:
        pickle.dump(data, fd, protocol=pickle.HIGHEST_PROTOCOL)
    print('Done!')
