from embeddings.sentence_level_glove_embedding import *
import numpy as np

import nltk
from nltk.corpus import stopwords

from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
import keras.layers.core
import keras.layers
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer

#########################################################################
# This file contains a sample lstm from:
# https://stackabuse.com/python-for-nlp-movie-sentiment-analysis-using-deep-learning-in-keras/
#########################################################################

def create_lstm(embedding_layer, binary_labels=True):
    # Create LSTM model
    model = Sequential()
    model.add(embedding_layer)
    model.add(LSTM(128))
    model.add(Dense(1 if binary_labels else 3, activation='sigmoid' if binary_labels else 'softmax'))
    model.compile(optimizer='adam', loss='binary_crossentropy' if binary_labels else 'categorical_crossentropy', metrics=['acc'])

    return model

def train_lstm(model, training_text, training_labels):
    history = model.fit(training_text, training_labels, batch_size=32, epochs=10, verbose=1, validation_split=0.1)
    return history

def test_lstm(model, testing_text, testing_labels, history):
    score = model.evaluate(testing_text, testing_labels, verbose=1)

    print("Test Score:", score[0])
    print("Test Accuracy:", score[1])

    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])

    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train','test'], loc = 'upper left')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])

    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train','test'], loc = 'upper left')
    plt.show()
