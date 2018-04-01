import json
from keras.preprocessing.text import Tokenizer,one_hot, text_to_word_sequence
from unidecode import unidecode
import cPickle as pickle
import numpy as np

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

import json

print('Reading SQuAD data... ')
with open("data/train_parsed.json") as fd:
    samples = json.load(fd)
print('Done!')

print len(samples)




