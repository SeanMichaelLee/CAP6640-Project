from embeddings.sentence_level_glove_embedding import *
import numpy as np

import nltk
from nltk.corpus import stopwords

from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers.core import Activation, Dropout, Dense
import keras.layers
from keras.layers.embeddings import Embedding
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.callbacks import ModelCheckpoint

#########################################################################
# This file contains a sample cnn from:
# https://stackabuse.com/python-for-nlp-movie-sentiment-analysis-using-deep-learning-in-keras/
#########################################################################

def create_cnn(embedding_layer, binary_labels=True):
    # Create CNN model
    model = Sequential()
    model.add(embedding_layer)
    model.add(keras.layers.Conv1D(128, 5, activation='relu'))
    model.add(keras.layers.GlobalMaxPooling1D())
    model.add(keras.layers.Dense(1 if binary_labels else 3, activation='sigmoid' if binary_labels else 'softmax'))
    model.compile(optimizer='adam', loss='binary_crossentropy' if binary_labels else 'categorical_crossentropy', metrics=['acc'])

    return model

def train_cnn(model, training_text, training_labels, filepath):
    callbacks = [ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)]
    history = model.fit(training_text, training_labels, batch_size=32, epochs=10, verbose=1, validation_split=0.111, callbacks=callbacks)
    return history

def test_cnn(model, testing_text, testing_labels, history):
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
