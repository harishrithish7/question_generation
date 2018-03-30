import json
import numpy as np
from pprint import pprint
from os import listdir
from os.path import dirname, abspath
from keras.preprocessing.text import Tokenizer,one_hot
import csv
import numpy as np
from keras.preprocessing.sequence import pad_sequences
import pickle
from sklearn.model_selection import StratifiedShuffleSplit
import re
import random

ROOT = dirname(dirname(abspath(__file__)))

DATA_DIR = ROOT + '/data/'

EMBEDDING_DIM = 300
VOCAB_LENGTH = 50000

def get_shape(x):
	s = []
	while type(x) == list:
		s.append(len(x))
		x = x[0]
	print s

def get_word_index(char_index, s):
	word_index = 0
	words = s.split()

	cur_len = 0
	for w in words:
		cur_len += len(w)
		if char_index < cur_len:
			return word_index

		cur_len += 1
		word_index += 1

	raise ValueError('Illegal char_index // buggy function')

def equals(x, y):
	x = ''.join(e for e in x if e.isalnum())
	y = ''.join(e for e in y if e.isalnum())
	return x == y

def read_file(fpath):
	data = json.loads(open(DATA_DIR + fpath).read())['data']

	paragraphs = []
	titles = []

	for passage_block in data:
		paragraphs.append(passage_block['paragraphs'])
		titles.append(passage_block['title'].strip('_'))

	contexts = []
	qas = []

	for para in paragraphs:
		context = []
		qa = []

		for context_block in para:
			context.append(context_block['context'].strip())
			qa.append(context_block['qas'])

		contexts.append(context)
		qas.append(qa)

	questions = []
	answers = []
	answers_start = []
	answers_end = []
	ids = []

	for context_block, qa in zip(contexts, qas):
		q_para = []
		a_para = []
		a_sn_para = []
		a_en_para = []
		id_para = []
		for context, q_a_id_triplets in zip(context_block, qa):
			q_context = []
			a_context = []
			a_sn_context = []
			a_en_context = []
			id_context = []

			for q_a_id_triplet in q_a_id_triplets:
				q_context.append(q_a_id_triplet['question'].strip())
				ass = q_a_id_triplet['answers']
				as_context = []
				as_sn_context = []
				as_en_context = [] 
				for a in ass:
					as_context.append(a['text'].strip())
					anss = get_word_index(int(a['answer_start']), context)
					as_sn_context.append(anss)
					en = anss + len(a['text'].strip().split()) - 1
					as_en_context.append(en)
					context_words = context.split()
					# if not equals(' '.join(context_words[idx] for idx in xrange(anss, en + 1, 1)), a['text'].strip()):
					# 	print context 
					# 	print a['text']
					# 	print anss, en
					# 	print ' '.join(context_words[idx] for idx in xrange(anss, en + 1, 1))
					# 	# if raw_input() == 'g':
					# 		continue
					# 	else: exit(0)
				a_context.append(as_context)
				a_sn_context.append(as_sn_context)
				a_en_context.append(as_en_context)
				id_context.append(q_a_id_triplet['id'])

			q_para.append(q_context)
			a_para.append(a_context)
			a_sn_para.append(a_sn_context)
			a_en_para.append(a_en_context)
			id_para.append(id_context)

		questions.append(q_para)
		answers.append(a_para)
		answers_start.append(a_sn_para)
		answers_end.append(a_en_para)
		ids.append(id_para)

	print 'Shape of contexts', np.asarray(contexts).shape
	print 'Shape of titles', np.asarray(titles).shape
	print 'Shape of questions', get_shape(questions)
	print 'Shape of answers', get_shape(answers)
	print 'Shape of ids', get_shape(ids)
	return preprocessed(titles),preprocessed(contexts),preprocessed(questions),preprocessed(answers),answers_start,answers_end,ids

def get_embeddings_index():
	with open(ROOT +'/glove/glove.840B.{}d.txt'.format(EMBEDDING_DIM)) as f:
		embeddings_index = {}
		for line in f:
			values = line.split(' ')
			word = values[0]
			coefs = np.asarray(values[1:], dtype='float32')
			embeddings_index[word] = coefs
	return embeddings_index

def flatten(x):
	if type(x[0]) != list: return x

	return flatten(sum(x, []))

def to_sequences(x, tokenizer):
	if type(x) != list:
		return tokenizer.texts_to_sequences([x])[0]

	for i in xrange(len(x)):
		x[i] = to_sequences(x[i], tokenizer)
	return x 

def preprocessed(x):
	if type(x) != list:
		if isinstance(x, unicode):
			x = x.encode('ascii', 'ignore')
			pre = []
			for word in x.strip().split():
				word = ''.join([c for c in word if c.isalnum()])
				word = re.sub('([0-9]+)', 'num', word)
				pre.append(word)
			x = ' '.join(pre)
		return x


	for i in xrange(len(x)):
		x[i] = preprocessed(x[i])

	return x 

def save_reverse_word_index(word_index):
	reverse_word_index = dict()
	for word, i in word_index.items():
		reverse_word_index[i] = word

	pickle.dump(reverse_word_index, open(DATA_DIR + 'reverse_word_index','wb'), protocol=pickle.HIGHEST_PROTOCOL)

def stringify(x):
	return [' '.join(s) for s in x]

def get_data():
	if 'len_word_index' in listdir(DATA_DIR):
		processed_data =  {'embedding_matrix':np.load(DATA_DIR + 'embedding_matrix.npy'),
			'len_word_index':pickle.load(open(DATA_DIR + 'len_word_index')),
			'x_train':np.load(DATA_DIR + 'x_train.npy'),
			'x_dev':np.load(DATA_DIR + 'x_dev.npy'),
			'x_val':np.load(DATA_DIR + 'x_val.npy'),
			'y_pos_train': np.load(DATA_DIR + 'y_pos_train.npy'),
			'y_pos_dev':np.load(DATA_DIR + 'y_pos_dev.npy'),
			'y_pos_val':np.load(DATA_DIR + 'y_pos_val.npy')
			}
		return processed_data

	tokenizer = Tokenizer(num_words=None, filters='')
	(contexts_train, questions_train), (start_train, end_train) = pickle.load(open(DATA_DIR + 'train_data_str.pkl', 'rb'))
	(contexts_val, questions_val), (start_val, end_val) = pickle.load(open(DATA_DIR + 'valid_data_str.pkl', 'rb'))
	(contexts_dev, questions_dev), (start_dev, end_dev) = pickle.load(open(DATA_DIR + 'dev_data_str.pkl', 'rb'))

	contexts_train, contexts_val, contexts_dev = stringify(contexts_train), stringify(contexts_val), stringify(contexts_dev)
	questions_train, questions_val, questions_dev = stringify(questions_train), stringify(questions_val), stringify(questions_dev)

	vocabulary = sum([contexts_train, contexts_val, contexts_dev, questions_train, questions_val, questions_dev], [])

	tokenizer.fit_on_texts(vocabulary)
	print "finished fitting"

	#train
	print 'Doing to_sequences for train...'

	xt_context = to_sequences(contexts_train, tokenizer)
	print 'Finsihed converting contexts to sequences'
	xt_question = to_sequences(questions_train, tokenizer)
	print 'Finsihed converting questions to sequences'
	x_train = [xt_context, xt_question]
	y_pos_train = start_train, end_train

	del xt_context,xt_question

	print "to sequences for train"

	#val
	print 'Doing to_sequences for val...'

	xv_context = to_sequences(contexts_val, tokenizer)
	print 'Finsihed converting contexts to sequences'
	xv_question = to_sequences(questions_val, tokenizer)
	print 'Finsihed converting questions to sequences'
	x_val = [xv_context, xv_question]
	y_pos_val = start_val, end_val

	del xv_context,xv_question

	print "to sequences for val"

	#dev
	print 'Doing to_sequences for dev...'

	xd_context = to_sequences(contexts_dev, tokenizer)
	print 'Finsihed converting contexts to sequences'
	xd_question = to_sequences(questions_dev, tokenizer)
	print 'Finsihed converting questions to sequences'
	x_dev = [xd_context, xd_question]
	y_pos_dev = start_dev, end_dev

	del xd_context,xd_question

	print "to sequences for dev"


	#get the embedding matrix

	word_index = tokenizer.word_index
	save_reverse_word_index(word_index)

	embeddings_index = get_embeddings_index()

	print "finished gloveing"
	embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
	for word, i in word_index.items():
		embedding_vector = embeddings_index.get(word)
		if embedding_vector is not None:
			embedding_matrix[i] = embedding_vector
		else:
			for j in range(0,EMBEDDING_DIM):
				embedding_matrix[i][j] = random.randint(-20000,+20000)/10000.0
	del embeddings_index
	print "finished gloaveing the vocab"

	pickle.dump(len(word_index), open(DATA_DIR + 'len_word_index', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
	np.save(DATA_DIR + 'embedding_matrix', embedding_matrix)
	np.save(DATA_DIR + 'x_train', np.asarray(x_train))
	np.save(DATA_DIR + 'x_dev', np.asarray(x_dev))
	np.save(DATA_DIR + 'x_val', np.asarray(x_val))
	np.save(DATA_DIR + 'y_pos_val', np.asarray(y_pos_val))
	np.save(DATA_DIR + 'y_pos_train', np.asarray(y_pos_train))
	np.save(DATA_DIR + 'y_pos_dev', np.asarray(y_pos_dev))

	processed_data =  {'embedding_matrix':embedding_matrix,
			'len_word_index':len(word_index),
			'x_train':x_train,
			'x_dev':x_dev,
			'y_train':x_val,
			'y_dev':y_pos_val,
			'y_pos_train': y_pos_train,
			'y_pos_dev':y_pos_dev
			}

	return processed_data

def get_text_from_indices(indices, reverse_word_index):
	if type(indices) == list: return ' '.join([reverse_word_index[idx] for idx in indices if idx != 0])
	return reverse_word_index[indices]

def data_visualizer(gen):
	reverse_word_index = pickle.load(open(DATA_DIR + 'reverse_word_index', 'rb'))

	go = True

	while go:
		X, y = gen.next()
		
		c, q = X
		anss, anse = y

		for context, question, answer_start, answer_end in zip(c, q, anss, anse):
			print get_text_from_indices(context.tolist(), reverse_word_index)
			print '\n'
			print get_text_from_indices(question.tolist(), reverse_word_index)
			print '\n'
			print get_text_from_indices(context.tolist()[np.argmax(answer_start):np.argmax(answer_end) + 1], reverse_word_index)
			print '\n'

			if raw_input() != 'g':
				break
		else:
			continue

		break

if __name__ == '__main__':
	d = get_data()


def old_get_data():
	if 'len_word_index' in listdir(DATA_DIR):
		processed_data =  {'embedding_matrix':np.load(DATA_DIR + 'embedding_matrix.npy'),
			'len_word_index':pickle.load(open(DATA_DIR + 'len_word_index')),
			'x_train':np.load(DATA_DIR + 'x_train.npy'),
			'x_dev':np.load(DATA_DIR + 'x_dev.npy'),
			'y_train':np.load(DATA_DIR + 'y_train.npy'),
			'y_dev':np.load(DATA_DIR + 'y_dev.npy'),
			'y_pos_train': np.load(DATA_DIR + 'y_pos_train.npy'),
			'y_pos_dev':np.load(DATA_DIR + 'y_pos_dev.npy')
			}
		return processed_data

	tokenizer = Tokenizer(num_words=VOCAB_LENGTH)
	titles_train, contexts_train, questions_train, answers_train, start_train, end_train, ids_train = read_file('train-v1.1.json')
	titles_dev, contexts_dev, questions_dev, answers_dev, start_dev, end_dev, ids_dev = read_file('dev-v1.1.json')


	if 'vocabulary' not in listdir(DATA_DIR):
		vocab_train = list(flatten(titles_train)) + list(flatten(contexts_train)) + list(flatten(questions_train)) + list(flatten(answers_train))
		print "flatten train"
		vocab_dev = list(flatten(titles_dev)) + list(flatten(contexts_dev)) + list(flatten(questions_dev)) + list(flatten(answers_dev))
		print "flatten dev"	

		vocabulary = vocab_train + vocab_dev
		pickle.dump(vocabulary, open(DATA_DIR + 'vocabulary','wb'), protocol=pickle.HIGHEST_PROTOCOL)
	else:
		vocabulary = pickle.load(open(DATA_DIR + 'vocabulary'))


	tokenizer.fit_on_texts(vocabulary)
	print "finished fitting"

	#train
	print 'Doing to_sequences for train...'

	xt_title = to_sequences(titles_train, tokenizer)
	print 'Finsihed converting titles to sequences'
	xt_context = to_sequences(contexts_train, tokenizer)
	print 'Finsihed converting contexts to sequences'
	xt_question = to_sequences(questions_train, tokenizer)
	print 'Finsihed converting questions to sequences'
	yt_answer = to_sequences(answers_train, tokenizer)
	print 'finished converting answers to sequences'
	x_train = [xt_title, xt_context, xt_question]
	y_train = yt_answer
	y_pos_train = start_train, end_train

	del xt_title,xt_context,xt_question
	# return x_train, y_train


	print "to sequences for train"

	#dev

	xd_title = to_sequences(titles_dev, tokenizer)
	xd_context = to_sequences(contexts_dev, tokenizer)
	xd_question = to_sequences(questions_dev, tokenizer)
	yd_answer = to_sequences(answers_dev, tokenizer)
	x_dev = [xd_title,xd_context,xd_question]
	y_dev = yd_answer
	y_pos_dev = start_dev, end_dev

	del xd_title,xd_context,xd_question
	print "to sequences for dev"


	#get the embedding matrix

	word_index = tokenizer.word_index
	save_reverse_word_index(word_index)

	embeddings_index = get_embeddings_index()

	print "finished gloveing"
	embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
	for word, i in word_index.items():
		embedding_vector = embeddings_index.get(word)
		if embedding_vector is not None:
			embedding_matrix[i] = embedding_vector
		else:
			for j in range(0,EMBEDDING_DIM):
				embedding_matrix[i][j] = random.randint(-20000,+20000)/10000.0
	del embeddings_index
	print "finished gloaveing the vocab"

	pickle.dump(len(word_index), open(DATA_DIR + 'len_word_index', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
	np.save(DATA_DIR + 'embedding_matrix', embedding_matrix)
	np.save(DATA_DIR + 'x_train', np.asarray(x_train))
	np.save(DATA_DIR + 'x_dev', np.asarray(x_dev))
	np.save(DATA_DIR + 'y_train', np.asarray(y_train))
	np.save(DATA_DIR + 'y_dev', np.asarray(y_dev))
	np.save(DATA_DIR + 'y_pos_train', np.asarray(y_pos_train))
	np.save(DATA_DIR + 'y_pos_dev', np.asarray(y_pos_dev))

	processed_data =  {'embedding_matrix':embedding_matrix,
			'len_word_index':len(word_index),
			'x_train':x_train,
			'x_dev':x_dev,
			'y_train':y_train,
			'y_dev':y_dev,
			'y_pos_train': y_pos_train,
			'y_pos_dev':y_pos_dev
			}

	# pickle.dump(processed_data, open(DATA_DIR + 'processed_data', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
	return processed_data