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
            return model[word.lower()]
        except KeyError:
            return model["<unk>"]
    return model, get_word_vector

def IntegerEncode(word, word_index):
    try:
        idx = word_index[word.lower()]
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

    print('Reading SQuAD data... ')
    with open(args.data) as fd:
        samples = json.load(fd)
    print('Done!')

    sNLP = StanfordNLP()

    word_vector, get_word_vector = word2vec(args.word2vec_path)

    word_index_map = dict((word,idx) for idx, word in enumerate(sorted(word_vector.vocab.keys())) )

    contexts = []
    qn_output = []
    qn_input = []
    answers = []

    # converts unicode to ascii
    print("Parsing samples")
    def parse_sample(sNLP, context, question, answer, **kwargs):
        tokens = sNLP.word_tokenize(unidecode(context))
        tokens = [token.lower() for token in tokens]
        contexts.append(tokens + ["<end>"])

        tokens = sNLP.word_tokenize(unidecode(question))
        tokens = [token.lower() for token in tokens]

        qn_input.append(["<start>"] + tokens)

        qn_out_int_enc = [IntegerEncode(token, word_index_map) for token in tokens] + [IntegerEncode("<end>", word_index_map)]
        qn_output.append(qn_out_int_enc)

        tokens = sNLP.word_tokenize(unidecode(answer))
        tokens = [token.lower() for token in tokens]
        answers.append(tokens)

        return None

    samples = [parse_sample(sNLP, **sample) for sample in tqdm(samples)]

    contexts, qn_output, answers, qn_input = np.array(contexts), np.array(qn_output), np.array(answers), np.array(qn_input)
    data = {
        "context": contexts,
        "qn_output": qn_output,
        "answer": answers,
        "qn_input": qn_input
    }

    print('Writing to file {}... '.format(args.outfile))
    with open(args.outfile, 'wb') as fd:
        pickle.dump(data, fd, protocol=pickle.HIGHEST_PROTOCOL)
    print('Done!')


    trimmed_len = 800
    indices = np.random.choice(len(contexts), trimmed_len, replace=False)

    data = {
        "context": contexts[indices],
        "qn_output": qn_output[indices],
        "answer": answers[indices],
        "qn_input": qn_input[indices]
    }

    print('Writing to file {}... '.format("data/preprocessed_data_trimmed.pkl"))
    with open("data/preprocessed_data_trimmed.pkl", 'wb') as fd:
        pickle.dump(data, fd, protocol=pickle.HIGHEST_PROTOCOL)
    print('Done!')
