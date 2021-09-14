import re
import matplotlib.pyplot as plt
import string
from nltk.corpus import stopwords
import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize.treebank import TreebankWordDetokenizer
from collections import Counter
from wordcloud import WordCloud
from nltk.corpus import stopwords
import nltk
from gensim.utils import simple_preprocess
from nltk.corpus import stopwords
import gensim
from sklearn.model_selection import train_test_split
import spacy
import pickle
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt 
import tensorflow as tf
import keras
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import wordnet 
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

print('imports done')
train = pd.read_csv('dataset/train.csv')
# textID, text, selected text, sentiment - selected text is the sentiment carrying part of the text
# dataset has positive, negative, neutral sentiments

train = train[['selected_text','sentiment']]

#drop rows with missing values
train = train.dropna()

whitespace = re.compile(r"\s+")
web_address = re.compile(r"(?i)http(s):\/\/[a-z0-9.~_\-\/]+")
#tesla = re.compile(r"(?i)@Tesla(?=\b)")
user = re.compile(r"(?i)@[a-z0-9_]+")
pic_pattern = re.compile('pic\.twitter\.com/.{10}')
special_code = re.compile(r'(&amp;|&gt;|&lt;)')
tag_pattern = re.compile(r'<.*?>')
emoji_pattern = re.compile("["
                        u"\U0001F600-\U0001F64F"  # emoticons
                        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                        u"\U0001F680-\U0001F6FF"  # transport & map symbols
                        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                        u"\U00002702-\U000027B0"
                        u"\U000024C2-\U0001F251"
                        "]+", flags=re.UNICODE)
emails = re.compile(r"\S*@\S*\s?")
single_quotes = re.compile(r"\'")
STOPWORDS = set(stopwords.words('english')).union(
    {'rt', 'retweet', 'RT', 'Retweet', 'RETWEET'})
lemmatizer = WordNetLemmatizer()

def tokenize_stem(phrase):
    tokens = word_tokenize(phrase)
    stem_words =[]
    for token in tokens:
        word = lemmatizer.lemmatize(token)
        stem_words.append(word)
    buf = ' '.join(stem_words)
    return buf

def remove_stopwords(phrase):
    return " ".join([word for word in str(phrase).split()\
                     if word not in STOPWORDS])

def clean_text(tweet):
	tweet = whitespace.sub(' ', tweet)
	tweet = web_address.sub('', tweet)
#	tweet = tesla.sub('Tesla', tweet)
	tweet = user.sub('', tweet)
	tweet = pic_pattern.sub('', tweet)
	tweet = special_code.sub('',tweet)
	tweet = tag_pattern.sub('',tweet)
	tweet = emoji_pattern.sub('',tweet)
	tweet = emails.sub('',tweet)
	tweet = single_quotes.sub('',tweet)
	tweet = tokenize_stem(tweet)
	tweet = remove_stopwords(tweet)
	return tweet

cleaned_text = []
df_to_list = train['selected_text'].values.tolist()
for i in range(len(df_to_list)):
	cleaned_text.append(clean_text(df_to_list[i]))

print('text cleaned')
text_words = []

def sentence_to_words(sentence):
	return gensim.utils.simple_preprocess(str(sentence), deacc=True)	

for sentence in cleaned_text:
	text_words.append(sentence_to_words(sentence))

def detokenize(text):
    return TreebankWordDetokenizer().detokenize(text)

text_detok = []
for i in range(len(text_words)):
	text_detok.append(detokenize(text_words[i]))

text_num = np.array(text_detok)
print('detokenized')

labels = np.array(train['sentiment'])
y = []
for i in range(len(labels)):
    if labels[i] == 'neutral':
        y.append(0)
    if labels[i] == 'negative':
        y.append(1)
    if labels[i] == 'positive':
        y.append(2)
y = np.array(y)
labels = tf.keras.utils.to_categorical(y, 3, dtype="float32")
print(len(labels))

from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop,Adam
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import regularizers
from keras import backend as K
from keras.callbacks import ModelCheckpoint
max_words = 5000
max_len = 200

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(text_num)
sequences = tokenizer.texts_to_sequences(text_num)
tweets = pad_sequences(sequences, maxlen=max_len)
print(tweets)

X_train, X_test, y_train, y_test = train_test_split(tweets,labels, random_state=0)
print (len(X_train),len(X_test),len(y_train),len(y_test))

model2 = Sequential()
model2.add(layers.Embedding(max_words, 40, input_length=max_len))
model2.add(layers.Bidirectional(layers.LSTM(20,dropout=0.6)))
model2.add(layers.Dense(3,activation='softmax'))
model2.compile(optimizer='rmsprop',loss='categorical_crossentropy', metrics=['accuracy'])
#Implementing model checkpoins to save the best metric and do not lose it on training.
checkpoint2 = ModelCheckpoint("best_model2.hdf5", monitor='val_accuracy', verbose=1,save_best_only=True, mode='auto', period=1,save_weights_only=False)
history = model2.fit(X_train, y_train, epochs=30,validation_data=(X_test, y_test),callbacks=[checkpoint2])
