from dataset import *

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import keras
from keras import backend as K
from keras import regularizers
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping
from keras.layers.embeddings import Embedding

from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
import os, re, csv, math, codecs
from urllib.request import urlopen
import gzip

def create_corpus():
    # Obtain dataset
    dataset = create_dataset()

    # Create the text list from the dataset
    text_list = []
    for index, row in dataset.iterrows():
        text_list.append(row['text'])

    # Create a list of labels from the dataset
    # NOTE: Positive = 2
    #       Neutral = 1
    #       Negative = 0
    labels = dataset['positivity']
    labels = np.array(list(map(lambda x: 2 if x=="Positive" else (0 if x=="Negative" else 1), labels)))

    # Divide the dataset 20% test and 80% training
    training_text, testing_text, training_labels, testing_labels = train_test_split(text_list, labels, test_size=0.20, random_state=42)

    # Create a word-to-index dictionary
    tokenizer = Tokenizer(num_words=10000)
    tokenizer.fit_on_texts(training_text)
    training_text = tokenizer.texts_to_sequences(training_text)
    testing_text = tokenizer.texts_to_sequences(testing_text)

    # Determine the vocabulary size and perform padding on the training and testing set
    vocab_size = len(tokenizer.word_index) + 1
    max_line_length = 1000
    training_text = pad_sequences(training_text, padding='post', maxlen=max_line_length)
    testing_text = pad_sequences(testing_text, padding='post', maxlen=max_line_length)

    # Load embedding
    embeddings_index = {}
    f = codecs.open('data/wiki.en.vec', encoding='utf-8')
    for line in tqdm(f):
      values = line.rstrip().rsplit(' ')
      word = values[0]
      coefs = np.asarray(values[1:], dtype='float32')
      embeddings_index[word] = coefs
    f.close()

    # Create a feature matrix using the fast tex pretrained embedding
    embedding_matrix = np.zeros((vocab_size, 300))
    for word, index in tokenizer.word_index.items():
      embedding_vector = embeddings_index.get(word)
      if (embedding_vector is not None) and len(embedding_vector) > 0:
          embedding_matrix[index] = embedding_vector

    embedding_layer = Embedding(vocab_size, 300, weights=[embedding_matrix], input_length=1000 , trainable=False)

    return training_text, testing_text, training_labels, testing_labels, embedding_layer
