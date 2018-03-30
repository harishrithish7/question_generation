# from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import json
import argparse
import random

if __name__ == '__main__':
    random.seed(42)

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, help='Path to the dataset file')
    parser.add_argument('--outfile_record', default='data/train_parsed_record.json',
                        type=str, help='Desired path to output train parsed record json')
    parser.add_argument('--outfile_valid_record', default='data/valid_parsed_record.json',
                        type=str, help='Desired path to output valid parsed record json')
    parser.add_argument('--outfile_sep', default='data/train_parsed_sep.json',
                        type=str, help='Desired path to output train parsed sep json')
    parser.add_argument('--outfile_valid_sep', default='data/valid_parsed_sep.json',
                        type=str, help='Desired path to output valid parsed sep json')
    parser.add_argument('--train_ratio', default=1., type=float,
                        help='ratio for train/val split')
    args = parser.parse_args()

    with open(args.data, 'r') as f:
        data = json.load(f)

    data = data['data']

    # Lists containing ContextQuestionAnswerS
    train_cqas_record = []
    valid_cqas_record = []

    # Lists containing ContextQuestionAnswerS, with context, ids, questions, etc.. all separate lists
    train_cqas_sep = {'context':      [],
                'id':           [],
                'question':     [],
                'answer':       [],
                'answer_start': [],
                'answer_end':   [],
                'topic':        [] }

    valid_cqas_sep = {'context':      [],
                'id':           [],
                'question':     [],
                'answer':       [],
                'answer_start': [],
                'answer_end':   [],
                'topic':        [] }



    for topic in data:
        cqas_record = [{'context':      paragraph['context'],
                'id':           qa['id'],
                'question':     qa['question'],
                'answer':       qa['answers'][0]['text'],
                'answer_start': qa['answers'][0]['answer_start'],
                'answer_end':   qa['answers'][0]['answer_start'] + \
                                len(qa['answers'][0]['text']) - 1,
                'topic':        topic['title'] }
                for paragraph in topic['paragraphs']
                for qa in paragraph['qas']]
           

        if random.random() < args.train_ratio:
            train_cqas_record += cqas_record
            for cqa in cqas_record:
                for key, value in cqa.iteritems():
                    train_cqas_sep[key].append(value)
        else:
            valid_cqas_record += cqas_record
            for cqa in cqas_record:
                for key, value in cqa.iteritems():
                    valid_cqas_sep[key].append(value)

    if args.train_ratio == 1.:
        print('Writing to file {}...'.format(args.outfile_record), end='')
        with open(args.outfile_record, 'w') as fd:
            json.dump(train_cqas_record, fd)
        print('Done!')

        print('Writing to file {}...'.format(args.outfile_sep), end='')
        with open(args.outfile_sep, 'w') as fd:
            json.dump(train_cqas_sep, fd)
        print('Done!')
    else:
        print('Train/Val ratio is {}'.format(len(train_cqas_record) / len(valid_cqas_record)))
        print('Writing to files {}, {}...'.format(args.outfile_record,
                                                  args.outfile_valid_record), end='')
        with open(args.outfile_record, 'w') as fd:
            json.dump(train_cqas_record, fd)
        with open(args.outfile_valid_record, 'w') as fd:
            json.dump(valid_cqas_record, fd)
        print('Done!')

        print('Writing to files {}, {}...'.format(args.outfile_sep,
                                                  args.outfile_valid_sep), end='')
        with open(args.outfile_sep, 'w') as fd:
            json.dump(train_cqas_sep, fd)
        with open(args.outfile_valid_sep, 'w') as fd:
            json.dump(valid_cqas_sep, fd)
        print('Done!')