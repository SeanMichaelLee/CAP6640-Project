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
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer

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
    labels = np.array(list(map(lambda x: 1 if x=="Positive" else 0, labels)))
    print(labels)

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

    return training_text, testing_text, training_labels, testing_labels, embedding_layer
