import spacy
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
from keras.preprocessing.sequence import pad_sequences
import os, re, csv, math, codecs
from urllib.request import urlopen
import gzip

max_line_length = 50

def aspect_level_format(training_text):
    # NOTE: Requires running python -m spacy download en_core_web_sm
    nlp = spacy.load('en_core_web_sm')
    training_text = map(lambda x:x.lower(),training_text)
    sentiment_terms = []
    for text in nlp.pipe(training_text):
            if text.is_parsed:
                sentiment_terms.append(' '.join([token.lemma_ for token in text if (not token.is_stop and not token.is_punct and (token.pos_ == "ADJ" or token.pos_ == "VERB"))]))
            else:
                sentiment_terms.append('')
    return sentiment_terms

def create_training_corpus(binary_labels=True):
    # Obtain dataset
    training_text, _, training_labels, _ = create_dataset(binary_labels=binary_labels)
    training_text = aspect_level_format(training_text)

    # Create a word-to-index dictionary
    tokenizer = Tokenizer(num_words=50000)
    tokenizer.fit_on_texts(training_text)
    training_text = tokenizer.texts_to_sequences(training_text)
    training_text = pad_sequences(training_text, padding='post', maxlen=max_line_length)

    return training_text, training_labels

def create_testing_corpus(binary_labels=True):
    # Obtain dataset
    training_text, testing_text, _, testing_labels = create_dataset(binary_labels=binary_labels)
    training_text = aspect_level_format(training_text)

    # Create a word-to-index dictionary
    tokenizer = Tokenizer(num_words=50000)
    tokenizer.fit_on_texts(training_text)
    testing_text = tokenizer.texts_to_sequences(testing_text)
    testing_text = pad_sequences(testing_text, padding='post', maxlen=max_line_length)

    return testing_text, testing_labels

def create_embedding_layer():
    # Obtain dataset
    training_text, _, training_labels, _ = create_dataset()
    training_text = aspect_level_format(training_text)

    # Create a word-to-index dictionary
    tokenizer = Tokenizer(num_words=50000)
    tokenizer.fit_on_texts(training_text)
    vocab_size = len(tokenizer.word_index) + 1
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

    embedding_layer = Embedding(vocab_size, 300, weights=[embedding_matrix], input_length=max_line_length , trainable=False)

    return embedding_layer
