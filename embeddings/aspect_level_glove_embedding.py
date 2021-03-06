import spacy
from dataset import *

import numpy as np

import nltk
from nltk.corpus import stopwords

from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers.core import Activation, Dropout, Dense
import keras.layers
from keras.layers.embeddings import Embedding
from keras.preprocessing.text import Tokenizer

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
    embeddings_dictionary = dict()
    glove_file = open('data/glove.6B.100d.txt', encoding="utf8")

    # Create a feature matrix using the GloVe embedding
    for line in glove_file:
        records = line.split()
        word = records[0]
        vector_dimensions = np.asarray(records[1:], dtype='float32')
        embeddings_dictionary [word] = vector_dimensions
    glove_file.close()

    # Create the embedding matrix
    embedding_matrix = np.zeros((vocab_size, 100))
    for word, index in tokenizer.word_index.items():
        embedding_vector = embeddings_dictionary.get(word)
        if embedding_vector is not None:
            embedding_matrix[index] = embedding_vector

    embedding_layer = Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=max_line_length , trainable=False)

    return embedding_layer
