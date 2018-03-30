import numpy as np
import json
import os
import argparse
import cPickle as pickle

from os import path
from gensim.scripts.glove2word2vec import glove2word2vec
from tqdm import tqdm
from unidecode import unidecode

from gensim.models import KeyedVectors

from stanfordcorenlp import StanfordCoreNLP
import logging

class StanfordNLP:
    def __init__(self, host='http://localhost', port=9000):
        self.nlp = StanfordCoreNLP(host, port=port,
                                   timeout=30000, quiet=True, logging_level=logging.WARNING)
        self.props = {
            'annotators': 'tokenize',
            'pipelineLanguage': 'en',
            'outputFormat': 'json'
        }

    def word_tokenize(self, sentence):
        return self.nlp.word_tokenize(sentence)


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

def get_word_index(word, word_index):
    try:
        idx = word_index[word]
    except KeyError:
        idx = word_index["<unk>"]
    return idx

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--word2vec_path', type=str,
                        default='data/glove.6B.100d.trimmed.vec',
                        help='Word2Vec vectors file path')
    parser.add_argument('--outfile', type=str, default='data/preprocessed_data.pkl',
                        help='Desired path to output pickle')
    parser.add_argument('--data', type=str, default="data/train_parsed_record.json",
                        help='Data json')
    args = parser.parse_args()

    if not args.outfile.endswith('.pkl'):
        args.outfile += '.pkl'

    word_vector, get_word_vector = word2vec(args.word2vec_path)
    print get_word_vector("The")
    print get_word_vector("the")
    print get_word_vector("<unk>")
