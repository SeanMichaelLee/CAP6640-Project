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

    # Divide the dataset 10% test and 90% training
    training_text, testing_text, training_labels, testing_labels = train_test_split(text_list, labels, test_size=0.10, random_state=42)

    # NOTE: Requires running python -m spacy download en_core_web_sm
    nlp = spacy.load('en_core_web_sm')
    training_text = map(lambda x:x.lower(),training_text)
    sentiment_terms = []
    for text in nlp.pipe(training_text):
            if text.is_parsed:
                sentiment_terms.append(' '.join([token.lemma_ for token in text if (not token.is_stop and not token.is_punct and (token.pos_ == "ADJ" or token.pos_ == "VERB"))]))
            else:
                sentiment_terms.append('')
    training_text = sentiment_terms

    # Create a word-to-index dictionary
    tokenizer = Tokenizer(num_words=50000)
    tokenizer.fit_on_texts(training_text)
    training_text = tokenizer.texts_to_sequences(training_text)
    testing_text = tokenizer.texts_to_sequences(testing_text)

    # Determine the vocabulary size and perform padding on the training and testing set
    vocab_size = len(tokenizer.word_index) + 1
    max_line_length = 1000
    training_text = pad_sequences(training_text, padding='post', maxlen=max_line_length)
    testing_text = pad_sequences(testing_text, padding='post', maxlen=max_line_length)

    embedding_layer = Embedding(vocab_size, 100, input_length=max_line_length)

    return training_text, testing_text, training_labels, testing_labels, embedding_layer
