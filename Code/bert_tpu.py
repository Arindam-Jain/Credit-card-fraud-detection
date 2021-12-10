import pandas as pd
import numpy as np

import tensorflow as tf
import transformers
import string
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import sklearn
from sklearn.metrics import confusion_matrix

import seaborn as sns
import matplotlib.pyplot as plt

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

import string
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

try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print('Running on TPU ', tpu.master())
except ValueError:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
else:
    strategy = tf.distribute.get_strategy()

print("REPLICAS: ", strategy.num_replicas_in_sync)

df = pd.read_csv('../input/smm-translated-train-and-test/train_translated.csv')
df.head()

print(f"The dataset contains { df.Label.nunique() } unique categories")

encoder = LabelEncoder()
df['categoryEncoded'] = encoder.fit_transform(df['Label'])

print(encoder.classes_)

df['translate_english'] = df['translate_english'].apply(lambda headline: str(headline).lower())
df['Claim'] = df['Claim'].apply(lambda descr: str(descr).lower())

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


df['descr_len'] = df['translate_english'].apply(lambda x: len(str(x).split()))
df['headline_len'] = df['Claim'].apply(lambda x: len(str(x).split()))

df.drop('Unnamed: 0', axis=1, inplace=True)

df[['Country (mentioned)','Claim','Source','translate_english','Label']]

df.head()

df.describe()

sns.distplot(df['descr_len'])
plt.title('Description Number of Words')
plt.show()

sns.distplot(df['headline_len'])
plt.title('Headline Number of Words')
plt.show()

df.head()

df["Country (mentioned)"].value_counts()
df['short_description'] = df['Country (mentioned)']+df['Claim'] + df['translate_english']
df['short_description'] = df['short_description'].map(lambda x: clean_text(x))
df['short_description'] .head()

df[['Country (mentioned)','Source','short_description','Label']]

df['short_description_len'] = df['short_description'].apply(lambda x: len(str(x).split()))

sns.distplot(df['short_description_len'])
plt.title('Short Description Number of Words')
plt.show()

def regular_encode(texts, tokenizer, maxlen=512):
    enc_di = tokenizer.batch_encode_plus(
        texts, 
#         return_attention_masks=False, 
        return_token_type_ids=False,
        pad_to_max_length=True,
        max_length=maxlen
    )
    
    return np.array(enc_di['input_ids'])

# %% [code] {"execution":{"iopub.status.busy":"2021-11-24T05:30:44.597463Z","iopub.execute_input":"2021-11-24T05:30:44.597916Z","iopub.status.idle":"2021-11-24T05:30:46.838010Z","shell.execute_reply.started":"2021-11-24T05:30:44.597883Z","shell.execute_reply":"2021-11-24T05:30:46.837056Z"}}
#bert large uncased pretrained tokenizer
tokenizer = transformers.BertTokenizer.from_pretrained('bert-large-uncased')

# %% [code] {"execution":{"iopub.status.busy":"2021-11-24T05:30:49.003469Z","iopub.execute_input":"2021-11-24T05:30:49.003747Z","iopub.status.idle":"2021-11-24T05:30:49.014029Z","shell.execute_reply.started":"2021-11-24T05:30:49.003720Z","shell.execute_reply":"2021-11-24T05:30:49.012826Z"}}
X_train,X_test ,y_train,y_test = train_test_split(df['short_description'], df['categoryEncoded'], random_state = 2020, test_size = 0.2)

# %% [code] {"execution":{"iopub.status.busy":"2021-11-24T05:30:51.046782Z","iopub.execute_input":"2021-11-24T05:30:51.047283Z","iopub.status.idle":"2021-11-24T05:32:05.989305Z","shell.execute_reply.started":"2021-11-24T05:30:51.047222Z","shell.execute_reply":"2021-11-24T05:32:05.988302Z"}}
#tokenizing the news descriptions and converting the categories into one hot vectors using tf.keras.utils.to_categorical
Xtrain_encoded = regular_encode(X_train.astype('str'), tokenizer, maxlen=128)
ytrain_encoded = tf.keras.utils.to_categorical(y_train, num_classes=4,dtype = 'int32')
Xtest_encoded = regular_encode(X_test.astype('str'), tokenizer, maxlen=128)
ytest_encoded = tf.keras.utils.to_categorical(y_test, num_classes=4,dtype = 'int32')

# %% [markdown]
# ## Building the model

# %% [code] {"execution":{"iopub.status.busy":"2021-11-24T05:32:05.991371Z","iopub.execute_input":"2021-11-24T05:32:05.991678Z","iopub.status.idle":"2021-11-24T05:32:06.000533Z","shell.execute_reply.started":"2021-11-24T05:32:05.991638Z","shell.execute_reply":"2021-11-24T05:32:05.999391Z"}}
def build_model(transformer, loss='categorical_crossentropy', max_len=512):
    input_word_ids = tf.keras.layers.Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
    sequence_output = transformer(input_word_ids)[0]
    cls_token = sequence_output[:, 0, :]
    #adding dropout layer
    x = tf.keras.layers.Dropout(0.3)(cls_token)
    #using a dense layer of 40 neurons as the number of unique categories is 40. 
    out = tf.keras.layers.Dense(4, activation='softmax')(x)
    model = tf.keras.Model(inputs=input_word_ids, outputs=out)
    #using categorical crossentropy as the loss as it is a multi-class classification problem
    model.compile(tf.keras.optimizers.Adam(lr=3e-5), loss=loss, metrics=['accuracy'])
    return model

# %% [code] {"execution":{"iopub.status.busy":"2021-11-24T05:32:06.002252Z","iopub.execute_input":"2021-11-24T05:32:06.002514Z","iopub.status.idle":"2021-11-24T05:34:00.222823Z","shell.execute_reply.started":"2021-11-24T05:32:06.002484Z","shell.execute_reply":"2021-11-24T05:34:00.221747Z"}}
#building the model on tpu
with strategy.scope():
    transformer_layer = transformers.TFAutoModel.from_pretrained('bert-large-uncased')
    model = build_model(transformer_layer, max_len=128)
model.summary()

# %% [markdown]
# ## Training

# %% [code] {"execution":{"iopub.status.busy":"2021-11-24T05:34:00.226124Z","iopub.execute_input":"2021-11-24T05:34:00.226375Z","iopub.status.idle":"2021-11-24T05:34:00.303566Z","shell.execute_reply.started":"2021-11-24T05:34:00.226347Z","shell.execute_reply":"2021-11-24T05:34:00.302646Z"}}
#creating the training and testing dataset.
BATCH_SIZE = 32*strategy.num_replicas_in_sync
AUTO = tf.data.experimental.AUTOTUNE 
train_dataset = (
    tf.data.Dataset
    .from_tensor_slices((Xtrain_encoded, ytrain_encoded))
    .repeat()
    .shuffle(2048)
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)
test_dataset = (
    tf.data.Dataset
    .from_tensor_slices(Xtest_encoded)
    .batch(BATCH_SIZE)
)

# %% [code] {"execution":{"iopub.status.busy":"2021-11-24T05:34:00.305070Z","iopub.execute_input":"2021-11-24T05:34:00.306028Z","iopub.status.idle":"2021-11-24T05:34:00.311990Z","shell.execute_reply.started":"2021-11-24T05:34:00.305988Z","shell.execute_reply":"2021-11-24T05:34:00.311067Z"}}
BATCH_SIZE

# %% [code] {"execution":{"iopub.status.busy":"2021-11-24T05:34:41.720621Z","iopub.execute_input":"2021-11-24T05:34:41.720963Z","iopub.status.idle":"2021-11-24T05:39:12.221914Z","shell.execute_reply.started":"2021-11-24T05:34:41.720913Z","shell.execute_reply":"2021-11-24T05:39:12.220822Z"}}
#training for 20 epochs
n_steps = Xtrain_encoded.shape[0] // BATCH_SIZE
train_history = model.fit(
    train_dataset,
    steps_per_epoch=n_steps,
    epochs=20
)

# %% [markdown]
# ## Evaluation

# %% [code] {"execution":{"iopub.status.busy":"2021-11-24T05:39:12.224007Z","iopub.execute_input":"2021-11-24T05:39:12.224338Z","iopub.status.idle":"2021-11-24T05:39:28.905091Z","shell.execute_reply.started":"2021-11-24T05:39:12.224297Z","shell.execute_reply":"2021-11-24T05:39:28.904066Z"}}
#making predictions
preds = model.predict(test_dataset,verbose = 1)
#converting the one hot vector output to a linear numpy array.
pred_classes = np.argmax(preds, axis = 1)

# %% [code] {"execution":{"iopub.status.busy":"2021-11-24T05:39:28.906652Z","iopub.execute_input":"2021-11-24T05:39:28.906916Z","iopub.status.idle":"2021-11-24T05:39:28.913285Z","shell.execute_reply.started":"2021-11-24T05:39:28.906888Z","shell.execute_reply":"2021-11-24T05:39:28.912458Z"}}
pred_classes

# %% [code] {"execution":{"iopub.status.busy":"2021-11-24T05:39:36.461372Z","iopub.execute_input":"2021-11-24T05:39:36.462276Z","iopub.status.idle":"2021-11-24T05:39:36.468078Z","shell.execute_reply.started":"2021-11-24T05:39:36.462232Z","shell.execute_reply":"2021-11-24T05:39:36.467343Z"}}
#extracting the classes from the label encoder
encoded_classes = encoder.classes_
#mapping the encoded output to actual categories
predicted_category = [encoded_classes[x] for x in pred_classes]
true_category = [encoded_classes[x] for x in y_test]

# %% [code] {"execution":{"iopub.status.busy":"2021-11-24T05:39:38.816283Z","iopub.execute_input":"2021-11-24T05:39:38.816810Z","iopub.status.idle":"2021-11-24T05:39:38.839703Z","shell.execute_reply.started":"2021-11-24T05:39:38.816778Z","shell.execute_reply":"2021-11-24T05:39:38.839048Z"}}
result_df = pd.DataFrame({'description':X_test,'true_category':true_category, 'predicted_category':predicted_category})
result_df.head()

# %% [code] {"execution":{"iopub.status.busy":"2021-11-24T05:39:45.534828Z","iopub.execute_input":"2021-11-24T05:39:45.535169Z","iopub.status.idle":"2021-11-24T05:39:45.550662Z","shell.execute_reply.started":"2021-11-24T05:39:45.535132Z","shell.execute_reply":"2021-11-24T05:39:45.549723Z"}}
result_df

# %% [code] {"execution":{"iopub.status.busy":"2021-11-24T05:40:01.109118Z","iopub.execute_input":"2021-11-24T05:40:01.109614Z","iopub.status.idle":"2021-11-24T05:40:01.118167Z","shell.execute_reply.started":"2021-11-24T05:40:01.109577Z","shell.execute_reply":"2021-11-24T05:40:01.117072Z"}}
print(f"Accuracy is {sklearn.metrics.accuracy_score(result_df['true_category'], result_df['predicted_category'])}")

# %% [code] {"execution":{"iopub.status.busy":"2021-11-23T21:13:56.580812Z","iopub.execute_input":"2021-11-23T21:13:56.581073Z","iopub.status.idle":"2021-11-23T21:13:56.668975Z","shell.execute_reply.started":"2021-11-23T21:13:56.581048Z","shell.execute_reply":"2021-11-23T21:13:56.668238Z"}}
result_df.to_csv('testPredictions.csv', index = False)

# %% [markdown]
# ## Confusion Matrix

# %% [code] {"execution":{"iopub.status.busy":"2021-11-24T05:40:16.218415Z","iopub.execute_input":"2021-11-24T05:40:16.218802Z","iopub.status.idle":"2021-11-24T05:40:16.238657Z","shell.execute_reply.started":"2021-11-24T05:40:16.218767Z","shell.execute_reply":"2021-11-24T05:40:16.237305Z"}}
result_df[result_df['true_category']!=result_df['predicted_category']]

# %% [code] {"execution":{"iopub.status.busy":"2021-11-24T05:40:21.659252Z","iopub.execute_input":"2021-11-24T05:40:21.659565Z","iopub.status.idle":"2021-11-24T05:40:21.673223Z","shell.execute_reply.started":"2021-11-24T05:40:21.659535Z","shell.execute_reply":"2021-11-24T05:40:21.671672Z"}}
confusion_mat = confusion_matrix(y_true = true_category, y_pred = predicted_category, labels=list(encoded_classes))

# %% [code] {"execution":{"iopub.status.busy":"2021-11-24T05:40:26.350755Z","iopub.execute_input":"2021-11-24T05:40:26.351102Z","iopub.status.idle":"2021-11-24T05:40:26.357693Z","shell.execute_reply.started":"2021-11-24T05:40:26.351051Z","shell.execute_reply":"2021-11-24T05:40:26.356850Z"}}
confusion_mat

# %% [code] {"execution":{"iopub.status.busy":"2021-11-24T05:41:17.664562Z","iopub.execute_input":"2021-11-24T05:41:17.665043Z","iopub.status.idle":"2021-11-24T05:41:18.083716Z","shell.execute_reply.started":"2021-11-24T05:41:17.664998Z","shell.execute_reply":"2021-11-24T05:41:18.082793Z"}}
df_cm = pd.DataFrame(confusion_mat, index = list(encoded_classes),columns = list(encoded_classes))
plt.rcParams['figure.figsize'] = (20,20)
sns.heatmap(df_cm)

# %% [code] {"execution":{"iopub.status.busy":"2021-11-24T05:41:21.964550Z","iopub.execute_input":"2021-11-24T05:41:21.964837Z","iopub.status.idle":"2021-11-24T05:41:21.977737Z","shell.execute_reply.started":"2021-11-24T05:41:21.964807Z","shell.execute_reply":"2021-11-24T05:41:21.977105Z"}}
df_cm

# %% [code] {"execution":{"iopub.status.busy":"2021-11-24T05:41:54.576015Z","iopub.execute_input":"2021-11-24T05:41:54.576496Z","iopub.status.idle":"2021-11-24T05:41:54.588335Z","shell.execute_reply.started":"2021-11-24T05:41:54.576461Z","shell.execute_reply":"2021-11-24T05:41:54.587331Z"}}
from sklearn.metrics import classification_report
#Show precision and recall per genre
print(classification_report(true_category, predicted_category ))

test_df = pd.read_csv('../input/smm-translated-train-and-test/test_translated.csv')
test_df.head()

# %% [code] {"execution":{"iopub.status.busy":"2021-11-24T05:43:23.426613Z","iopub.execute_input":"2021-11-24T05:43:23.426962Z","iopub.status.idle":"2021-11-24T05:43:23.445712Z","shell.execute_reply.started":"2021-11-24T05:43:23.426914Z","shell.execute_reply":"2021-11-24T05:43:23.445036Z"}}
test_df['translate_english'] = test_df['translate_english'].apply(lambda headline: str(headline).lower())
test_df['Claim'] = test_df['Claim'].apply(lambda descr: str(descr).lower())
test_df['short_description'] = test_df['Claim'] + test_df['translate_english']

# %% [code] {"execution":{"iopub.status.busy":"2021-11-24T05:43:24.076467Z","iopub.execute_input":"2021-11-24T05:43:24.077157Z","iopub.status.idle":"2021-11-24T05:43:24.804122Z","shell.execute_reply.started":"2021-11-24T05:43:24.077116Z","shell.execute_reply":"2021-11-24T05:43:24.803367Z"}}
test_df['short_description'] = test_df['short_description'].map(lambda x: clean_text(x))

# %% [code] {"execution":{"iopub.status.busy":"2021-11-24T05:43:26.110504Z","iopub.execute_input":"2021-11-24T05:43:26.111277Z","iopub.status.idle":"2021-11-24T05:43:26.115778Z","shell.execute_reply.started":"2021-11-24T05:43:26.111237Z","shell.execute_reply":"2021-11-24T05:43:26.114763Z"}}
XX_test=test_df['short_description']


# %% [code] {"execution":{"iopub.status.busy":"2021-11-24T05:43:28.965625Z","iopub.execute_input":"2021-11-24T05:43:28.965911Z","iopub.status.idle":"2021-11-24T05:43:28.972700Z","shell.execute_reply.started":"2021-11-24T05:43:28.965881Z","shell.execute_reply":"2021-11-24T05:43:28.972072Z"}}
XX_test.head()

# %% [code] {"execution":{"iopub.status.busy":"2021-11-24T05:43:29.790393Z","iopub.execute_input":"2021-11-24T05:43:29.790882Z","iopub.status.idle":"2021-11-24T05:43:37.900083Z","shell.execute_reply.started":"2021-11-24T05:43:29.790842Z","shell.execute_reply":"2021-11-24T05:43:37.899036Z"}}
XXtest_encoded = regular_encode(XX_test.astype('str'), tokenizer, maxlen=128)

# %% [code] {"execution":{"iopub.status.busy":"2021-11-24T05:43:37.901845Z","iopub.execute_input":"2021-11-24T05:43:37.902089Z","iopub.status.idle":"2021-11-24T05:43:37.906810Z","shell.execute_reply.started":"2021-11-24T05:43:37.902062Z","shell.execute_reply":"2021-11-24T05:43:37.906077Z"}}
XXtest_encoded.shape

# %% [code] {"execution":{"iopub.status.busy":"2021-11-24T05:43:37.907880Z","iopub.execute_input":"2021-11-24T05:43:37.908126Z","iopub.status.idle":"2021-11-24T05:43:37.929962Z","shell.execute_reply.started":"2021-11-24T05:43:37.908098Z","shell.execute_reply":"2021-11-24T05:43:37.929222Z"}}
ttest_dataset = (
    tf.data.Dataset
    .from_tensor_slices(XXtest_encoded)
    .batch(BATCH_SIZE)
)

ppreds = model.predict(ttest_dataset,verbose = 1)
#converting the one hot vector output to a linear numpy array.
ppred_classes = np.argmax(ppreds, axis = 1)

# %% [code] {"execution":{"iopub.status.busy":"2021-11-24T05:43:47.136075Z","iopub.execute_input":"2021-11-24T05:43:47.136427Z","iopub.status.idle":"2021-11-24T05:43:47.147755Z","shell.execute_reply.started":"2021-11-24T05:43:47.136385Z","shell.execute_reply":"2021-11-24T05:43:47.146818Z"}}
ppreds[:5]

# %% [code] {"execution":{"iopub.status.busy":"2021-11-24T05:48:03.336591Z","iopub.execute_input":"2021-11-24T05:48:03.337202Z","iopub.status.idle":"2021-11-24T05:48:03.352217Z","shell.execute_reply.started":"2021-11-24T05:48:03.337163Z","shell.execute_reply":"2021-11-24T05:48:03.351235Z"}}
sub_df=pd.read_csv('/kaggle/input/smmfall21asu/sample_submission.csv')
bert_sub = sub_df.copy()
df_bert = pd.DataFrame(ppred_classes, columns = ['Predicted'])
bert_sub['Predicted']= df_bert['Predicted']
bert_sub["Predicted"].replace({3:0,1:0}, inplace=True)
bert_sub.to_csv('bert_final.csv',index=False)

