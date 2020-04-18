from __future__ import division, print_function

from embeddings.sentence_level_glove_embedding import *

from gensim import models
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Dropout, Reshape, Flatten, concatenate, Input, Conv1D, GlobalMaxPooling1D, Embedding, Activation
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import os
import collections
import re
import string

import numpy as np
import keras
import pandas as pd

#########################################################################
# This file contains a sample lstm from:
# https://stackabuse.com/python-for-nlp-movie-sentiment-analysis-using-deep-learning-in-keras/
#########################################################################

def create_lstm(embedding_layer, embedding_dim, labels_index, binary_labels=True):

  model = Sequential()
  model.add(embedding_layer)
  model.add(LSTM(embedding_dim, return_sequences=True))
  model.add(Dense(units=128, input_shape=(10412,), activation='relu'))
  model.add(Flatten())
  model.add(Dropout(0.5))
  model.add(Dense(units=1 if binary_labels else 3, input_shape=(128,), activation='sigmoid' if binary_labels else 'softmax'))
  model.compile(loss='binary_crossentropy' if binary_labels else 'categorical_crossentropy', optimizer='adam', metrics=['acc'])
  model.summary()

  return model

def train_lstm(model, training_text, training_labels):
    training_text
    training_labels
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
