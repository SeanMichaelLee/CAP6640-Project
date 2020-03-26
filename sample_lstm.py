from embeddings.glove_embedding import *
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

def create_lstm(embedding_layer):
    # Create LSTM model
    model = Sequential()
    model.add(embedding_layer)
    model.add(LSTM(128))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

    return model

def train_lstm(model, training_text, training_labels):
    history = model.fit(training_text, training_labels, batch_size=128, epochs=10, verbose=1, validation_split=0.2)
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

training_text, testing_text, training_labels, testing_labels, embedding_layer = create_corpus()
model = create_lstm(embedding_layer)
history = train_lstm(model, training_text, training_labels)

# Save model
model.save("models/lstm_model.h5")
print("Saved model to disk")

test_lstm(model, testing_text, testing_labels, history)
