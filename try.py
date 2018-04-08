import json
from keras.preprocessing.text import Tokenizer,one_hot, text_to_word_sequence
from unidecode import unidecode
import cPickle as pickle
import numpy as np
from keras.models import Model
from model import PredictionDecoderModel

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

"""import json

print('Reading SQuAD data... ')
with open("data/train_parsed_trimmed.pkl") as fd:
    samples = pickle.load(f)
print('Done!')

print len(samples)

model = TrainingModel()
model.load_weights("pre-trained/current.hdf5")"""

model = PredictionDecoderModel()




