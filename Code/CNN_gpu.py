import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
import tensorflow as tf
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from tensorflow import keras
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras import layers
import string
from keras.callbacks import EarlyStopping
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import *
from tensorflow.keras.models import *

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import tensorflow_datasets as tfds

import warnings
warnings.filterwarnings('ignore')

train_data = pd.read_csv('../input/smm-translated-train-and-test/train_translated.csv')
test_data = pd.read_csv('../input/smm-translated-train-and-test/test_translated.csv')

# %% [code] {"id":"iSz_Lms7N65A","outputId":"1911a0e9-9cec-40bb-bfe8-a497925356c9","execution":{"iopub.status.busy":"2021-11-24T04:49:29.817936Z","iopub.execute_input":"2021-11-24T04:49:29.818796Z","iopub.status.idle":"2021-11-24T04:49:29.824381Z","shell.execute_reply.started":"2021-11-24T04:49:29.818745Z","shell.execute_reply":"2021-11-24T04:49:29.823503Z"},"jupyter":{"outputs_hidden":false}}
print(f"Number of sentences in train data {len(train_data)}")
print(f"Number of sentences in test data {len(test_data)}")

train_data.head()

def clean_text(text):
    
    ## Remove puncuation
    text = text.translate(string.punctuation)
    
    ## Convert words to lower case and split them
    text = text.lower().split()
    
    ## Remove stop words
    stops = set(stopwords.words("english"))
    text = [w for w in text if not w in stops and len(w) >= 3]
    
    text = " ".join(text)
    ## Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)
    ## Stemming
    text = text.split()
    stemmer = PorterStemmer()
    stemmed_words = [stemmer.stem(word) for word in text]
    text = " ".join(stemmed_words)
    return text

# %% [code] {"id":"vZarLLCtOky_","execution":{"iopub.status.busy":"2021-11-24T04:49:50.777608Z","iopub.execute_input":"2021-11-24T04:49:50.777888Z","iopub.status.idle":"2021-11-24T04:50:45.371244Z","shell.execute_reply.started":"2021-11-24T04:49:50.777858Z","shell.execute_reply":"2021-11-24T04:50:45.370441Z"},"jupyter":{"outputs_hidden":false}}
train_data['translate_english'] = train_data['translate_english'].map(lambda x: clean_text(x))
test_data['translate_english'] = test_data['translate_english'].map(lambda x: clean_text(x))

# %% [code] {"id":"-uahTbaQO5do","outputId":"8294a937-8a2b-4036-cdb1-53b9a314b7a5","execution":{"iopub.status.busy":"2021-11-24T04:50:45.372826Z","iopub.execute_input":"2021-11-24T04:50:45.373056Z","iopub.status.idle":"2021-11-24T04:50:45.391181Z","shell.execute_reply.started":"2021-11-24T04:50:45.373024Z","shell.execute_reply":"2021-11-24T04:50:45.390175Z"},"jupyter":{"outputs_hidden":false}}
train_data.head()

# %% [code] {"id":"B22VwB6yO-pL","execution":{"iopub.status.busy":"2021-11-24T04:50:45.392870Z","iopub.execute_input":"2021-11-24T04:50:45.393168Z","iopub.status.idle":"2021-11-24T04:50:45.397934Z","shell.execute_reply.started":"2021-11-24T04:50:45.393118Z","shell.execute_reply":"2021-11-24T04:50:45.396946Z"},"jupyter":{"outputs_hidden":false}}
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# %% [code] {"id":"5ymLEztsPOMW","execution":{"iopub.status.busy":"2021-11-24T04:50:45.400171Z","iopub.execute_input":"2021-11-24T04:50:45.400703Z","iopub.status.idle":"2021-11-24T04:50:45.410540Z","shell.execute_reply.started":"2021-11-24T04:50:45.400666Z","shell.execute_reply":"2021-11-24T04:50:45.409866Z"},"jupyter":{"outputs_hidden":false}}
sentences_train=train_data["translate_english"].values
y_train=train_data["Label"].values
sentences_test=test_data["translate_english"].values
# y_test=test_data["Label"].values

# %% [code] {"execution":{"iopub.status.busy":"2021-11-24T04:50:45.411972Z","iopub.execute_input":"2021-11-24T04:50:45.412759Z","iopub.status.idle":"2021-11-24T04:50:54.529189Z","shell.execute_reply.started":"2021-11-24T04:50:45.412723Z","shell.execute_reply":"2021-11-24T04:50:54.527479Z"},"jupyter":{"outputs_hidden":false}}
def load_glove_vectors(filepath):
    
    model = {}
    print("Loading Glove Model")
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            words = line.split()
            model[words[0]] = np.array([float(value) for value in words[1:]])
    print("Total loaded words", len(model))
    
    return model

glove_model = load_glove_vectors("../input/glove6b50dtxt/glove.6B.50d.txt")

# %% [code] {"id":"WMWYClmkQr5y","execution":{"iopub.status.busy":"2021-11-24T04:50:54.530901Z","iopub.execute_input":"2021-11-24T04:50:54.531699Z","iopub.status.idle":"2021-11-24T04:50:58.964236Z","shell.execute_reply.started":"2021-11-24T04:50:54.531631Z","shell.execute_reply":"2021-11-24T04:50:58.963440Z"},"jupyter":{"outputs_hidden":false}}
tokenizer=Tokenizer(num_words=5000)
tokenizer.fit_on_texts(sentences_train)

X_train=tokenizer.texts_to_sequences(sentences_train)
X_test=tokenizer.texts_to_sequences(sentences_test)

vocab_size=len(tokenizer.word_index)+1

maxlen=100

X_train=pad_sequences(X_train,padding='post',maxlen=maxlen)
X_test=pad_sequences(X_test,padding='post',maxlen=maxlen)

def create_embedding_matrix(filepath,word_index,embedding_dim):
    vocab_size=len(word_index)+1
    embedding_matrix=np.zeros((vocab_size,embedding_dim))
    with open(filepath,encoding="utf8") as f:
        for line in f:
            word,*vector=line.split()
            if word in word_index:
                idx=word_index[word]
                embedding_matrix[idx]=np.array(vector,dtype=np.float32)[:embedding_dim]
    return embedding_matrix

embedding_dim=50
embedding_matrix=create_embedding_matrix('../input/glove6b50dtxt/glove.6B.50d.txt',tokenizer.word_index,embedding_dim)

# %% [code] {"id":"22wDLpunQ17I","outputId":"2f03dfb6-7f8a-45d2-8668-4d912acc9c12","execution":{"iopub.status.busy":"2021-11-24T04:56:14.346681Z","iopub.execute_input":"2021-11-24T04:56:14.347238Z","iopub.status.idle":"2021-11-24T04:56:25.341481Z","shell.execute_reply.started":"2021-11-24T04:56:14.347202Z","shell.execute_reply":"2021-11-24T04:56:25.340704Z"},"jupyter":{"outputs_hidden":false}}
from keras.models import Sequential
from keras import layers
embedding_dim = 100

model = Sequential()
model.add(layers.Embedding(vocab_size, embedding_dim, input_length=maxlen))
model.add(layers.Conv1D(128, 5, activation='relu'))
model.add(layers.GlobalMaxPooling1D())
model.add(layers.Dense(10, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)
history = model.fit(X_train, y_train,
                    epochs=20,
                    batch_size=128,callbacks=[es])
y_pred = model.predict(X_train, batch_size=64, verbose=1)
y_pred_bool = np.argmax(y_pred, axis=1)
print(classification_report(y_train, y_pred.round()))
report1=classification_report(y_train, y_pred.round(),output_dict=True)

# %% [code] {"execution":{"iopub.status.busy":"2021-11-24T04:56:30.625547Z","iopub.execute_input":"2021-11-24T04:56:30.626092Z","iopub.status.idle":"2021-11-24T04:56:30.631877Z","shell.execute_reply.started":"2021-11-24T04:56:30.626054Z","shell.execute_reply":"2021-11-24T04:56:30.631202Z"},"jupyter":{"outputs_hidden":false}}
from sklearn.metrics import mean_squared_error
mean_squared_error(y_pred_bool, y_train)

# %% [code] {"execution":{"iopub.status.busy":"2021-11-24T04:52:36.060151Z","iopub.execute_input":"2021-11-24T04:52:36.060928Z","iopub.status.idle":"2021-11-24T04:52:36.067441Z","shell.execute_reply.started":"2021-11-24T04:52:36.060874Z","shell.execute_reply":"2021-11-24T04:52:36.066641Z"},"jupyter":{"outputs_hidden":false}}
y_pred_bool

# %% [code] {"id":"U-JinxQ_U9Dv","outputId":"21307016-97d3-4767-8e7b-c8d43aa8a1bf","execution":{"iopub.status.busy":"2021-11-24T04:52:38.957347Z","iopub.execute_input":"2021-11-24T04:52:38.957877Z","iopub.status.idle":"2021-11-24T04:52:39.152250Z","shell.execute_reply.started":"2021-11-24T04:52:38.957840Z","shell.execute_reply":"2021-11-24T04:52:39.151566Z"},"jupyter":{"outputs_hidden":false}}
plt.plot(history.history['loss'], label='train')
# plt.plot(history.history['val_loss'], label='val')
plt.legend()
plt.show()

# %% [code] {"execution":{"iopub.status.busy":"2021-11-24T04:52:43.910192Z","iopub.execute_input":"2021-11-24T04:52:43.910453Z","iopub.status.idle":"2021-11-24T04:52:43.971625Z","shell.execute_reply.started":"2021-11-24T04:52:43.910423Z","shell.execute_reply":"2021-11-24T04:52:43.970974Z"},"jupyter":{"outputs_hidden":false}}
y_pred = model.predict(X_test, batch_size=64, verbose=1)
y_pred_bool = np.argmax(y_pred, axis=1)

# %% [code] {"jupyter":{"outputs_hidden":false}}
