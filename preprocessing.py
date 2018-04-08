import numpy as np
import json
import argparse
import cPickle as pickle
from os import path
from tqdm import tqdm
from unidecode import unidecode
from gensim.models import KeyedVectors
from stanford_corenlp_pywrapper import CoreNLP
from word2vec_preprocessing import embedding_dimension, corpus

CoreNLP_path = '/home/h7predator/ml_code/stanford-corenlp-full-2018-02-27/'

contexts = []
qn_output = []
qn_input = []
answers = []
sentences = []

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

def CoreNLP_tokenizer():
    proc = CoreNLP(configdict={'annotators': 'tokenize,ssplit'},
                   corenlp_jars=[path.join(CoreNLP_path, '*')])

    def tokenize_with_offset(context):
        parsed = proc.parse_doc(context)
        return [(sentence['tokens'], sentence['char_offsets'][0][0], sentence['char_offsets'][-1][-1])
                     for sentence in parsed['sentences']]

    def tokenize(sentence):
        parsed = proc.parse_doc(sentence)
        tokens = []
        for sentence in parsed['sentences']:
            tokens += sentence['tokens']
        return tokens

    return tokenize_with_offset, tokenize

def parse_sample(tokenize_with_offset, tokenize, word_index_map, context, question, answer, answer_start, answer_end, **kwargs):
    context, question, answer = unidecode(context).lower(), unidecode(question).lower(), unidecode(answer).lower()

    ans_start,ans_end = answer_start,answer_end
    parsed_cxt = tokenize_with_offset(context)
    sentence = []

    buff = 1
    for sent_obj in parsed_cxt:
        sent, sent_start, sent_end = sent_obj
        if sent_start >= ans_start and sent_start+buff < ans_end:
            sentence += sent
        elif sent_start < ans_start and ans_start < sent_end-buff:
            sentence += sent
        if sent_start+buff >= answer_end:
            break

    if not len(answer) or not len(sentence):
        return None

    sentences.append(sentence + ["<end>"])

    tokens = tokenize(context)
    tokens = [unidecode(token) for token in tokens]
    contexts.append(tokens + ["<end>"])

    tokens = tokenize(question)
    tokens = [unidecode(token) for token in tokens]

    qn_input.append(["<start>"] + tokens)

    qn_out_int_enc = [IntegerEncode(token, word_index_map) for token in tokens] + [IntegerEncode("<end>", word_index_map)]
    qn_output.append(qn_out_int_enc)

    tokens = tokenize(answer)
    tokens = [unidecode(token) for token in tokens]
    answers.append(tokens)

    return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--word2vec_path', type=str,
                        default='data/glove.{}B.{}d.decoder.vec'.format(corpus, embedding_dimension),
                        help='Word2Vec vectors file path')
    parser.add_argument('--outfile', type=str, default='data/preprocessed_data.pkl',
                        help='Desired path to output pickle')
    parser.add_argument('--data', type=str, default="data/train_parsed.json",
                        help='Data json')
    args = parser.parse_args()

    if not args.outfile.endswith('.pkl'):
        args.outfile += '.pkl'

    print('Reading SQuAD data... ')
    with open(args.data) as fd:
        samples = json.load(fd)
    print('Done!')

    tokenize_with_offset, tokenize = CoreNLP_tokenizer()

    word_vector, get_word_vector = word2vec(args.word2vec_path)
    word_index_map = dict((word,idx) for idx, word in enumerate(sorted(word_vector.vocab.keys())) )

    print("Parsing samples")
    samples = [parse_sample(tokenize_with_offset, tokenize, word_index_map, **sample) for sample in tqdm(samples)]
    print ("Done!")

    sentences, contexts, qn_output, answers, qn_input = np.array(sentences), np.array(contexts), np.array(qn_output), np.array(answers), np.array(qn_input)
    data = {
        "sentence": sentences,
        "context": contexts,
        "qn_output": qn_output,
        "answer": answers,
        "qn_input": qn_input
    }

    print('Writing to file {}... '.format(args.outfile))
    with open(args.outfile, 'wb') as fd:
        pickle.dump(data, fd, protocol=pickle.HIGHEST_PROTOCOL)
    print('Done!')


    trimmed_len = 1000
    indices = np.random.choice(len(contexts), trimmed_len, replace=False)

    data = {
        "sentence": sentences[indices],
        "context": contexts[indices],
        "qn_output": qn_output[indices],
        "answer": answers[indices],
        "qn_input": qn_input[indices]
    }

    print('Writing to file {}... '.format("data/preprocessed_data_trimmed.pkl"))
    with open("data/preprocessed_data_trimmed.pkl", 'wb') as fd:
        pickle.dump(data, fd, protocol=pickle.HIGHEST_PROTOCOL)
    print('Done!')

