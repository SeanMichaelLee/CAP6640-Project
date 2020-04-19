import embeddings.aspect_level_fast_text_embedding
import embeddings.aspect_level_glove_embedding
import embeddings.aspect_level_word2vec_embedding
import embeddings.sentence_level_fast_text_embedding
import embeddings.sentence_level_glove_embedding
import embeddings.sentence_level_word2vec_embedding
import cnn
import lstm
import deep_lstm

import numpy as np

def train_cnn_model(embedding_layer, training_text, training_labels, name, binary_labels=True):
    model = cnn.create_cnn(embedding_layer, binary_labels=binary_labels)
    history = cnn.train_cnn(model, training_text, training_labels, "models/{}.h5".format(name))

    print("Saved {} to disk".format(name))

def train_lstm_model(embedding_layer, training_text, training_labels, name, binary_labels=True):
    model = lstm.create_lstm(embedding_layer, binary_labels=binary_labels)
    history = lstm.train_lstm(model, training_text, training_labels, "models/{}.h5".format(name))

    print("Saved {} to disk".format(name))

def train_deep_lstm_model(embedding_layer, training_text, training_labels, name, binary_labels=True):
    model = deep_lstm.create_lstm(embedding_layer, 1000, 2, binary_labels=binary_labels)
    history = deep_lstm.train_lstm(model, training_text, training_labels, "models/{}.h5".format(name))

    print("Saved {} to disk".format(name))

# Train binary classification models

embedding_layer = embeddings.aspect_level_fast_text_embedding.create_embedding_layer()
training_text, training_labels = embeddings.aspect_level_fast_text_embedding.create_training_corpus()
train_cnn_model(embedding_layer, training_text, training_labels, "cnn_aspect_level_fast_text")
train_lstm_model(embedding_layer, training_text, training_labels, "lstm_aspect_level_fast_text")
train_deep_lstm_model(embedding_layer, training_text, training_labels, "deep_lstm_aspect_level_fast_text")

embedding_layer = embeddings.aspect_level_glove_embedding.create_embedding_layer()
training_text, training_labels = embeddings.aspect_level_glove_embedding.create_training_corpus()
train_cnn_model(embedding_layer, training_text, training_labels, "cnn_aspect_level_glove")
train_lstm_model(embedding_layer, training_text, training_labels, "lstm_aspect_level_glove")
train_deep_lstm_model(embedding_layer, training_text, training_labels, "deep_lstm_aspect_level_glove")

embedding_layer = embeddings.aspect_level_word2vec_embedding.create_embedding_layer()
training_text, training_labels = embeddings.aspect_level_word2vec_embedding.create_training_corpus()
train_cnn_model(embedding_layer, training_text, training_labels, "cnn_aspect_level_word2vec")
train_lstm_model(embedding_layer, training_text, training_labels, "lstm_aspect_level_word2vec")
train_deep_lstm_model(embedding_layer, training_text, training_labels, "deep_lstm_aspect_level_word2vec")

embedding_layer = embeddings.sentence_level_fast_text_embedding.create_embedding_layer()
training_text, training_labels = embeddings.sentence_level_fast_text_embedding.create_training_corpus()
train_cnn_model(embedding_layer, training_text, training_labels, "cnn_sentence_level_fast_text")
train_lstm_model(embedding_layer, training_text, training_labels, "lstm_sentence_level_fast_text")
train_deep_lstm_model(embedding_layer, training_text, training_labels, "deep_lstm_sentence_level_fast_text")

embedding_layer = embeddings.sentence_level_glove_embedding.create_embedding_layer()
training_text, training_labels = embeddings.sentence_level_glove_embedding.create_training_corpus()
train_cnn_model(embedding_layer, training_text, training_labels, "cnn_sentence_level_fast_text")
train_lstm_model(embedding_layer, training_text, training_labels, "lstm_sentence_level_fast_text")
train_deep_lstm_model(embedding_layer, training_text, training_labels, "deep_lstm_sentence_level_fast_text")

embedding_layer = embeddings.sentence_level_word2vec_embedding.create_embedding_layer()
training_text, training_labels = embeddings.sentence_level_word2vec_embedding.create_training_corpus()
train_cnn_model(embedding_layer, training_text, training_labels, "cnn_sentence_level_fast_text")
train_lstm_model(embedding_layer, training_text, training_labels, "lstm_sentence_level_fast_text")
train_deep_lstm_model(embedding_layer, training_text, training_labels, "deep_lstm_sentence_level_fast_text")

# Train multi-class models

embedding_layer = embeddings.aspect_level_fast_text_embedding.create_embedding_layer()
training_text, training_labels = embeddings.aspect_level_fast_text_embedding.create_training_corpus(binary_labels=False)
train_cnn_model(embedding_layer, training_text, training_labels, "cnn_aspect_level_fast_text_3", binary_labels=False)
train_lstm_model(embedding_layer, training_text, training_labels, "lstm_aspect_level_fast_text_3", binary_labels=False)
train_deep_lstm_model(embedding_layer, training_text, training_labels, "deep_lstm_aspect_level_fast_text_3", binary_labels=False)

embedding_layer = embeddings.aspect_level_glove_embedding.create_embedding_layer()
training_text, training_labels = embeddings.aspect_level_glove_embedding.create_training_corpus(binary_labels=False)
train_cnn_model(embedding_layer, training_text, training_labels, "cnn_aspect_level_glove_3", binary_labels=False)
train_lstm_model(embedding_layer, training_text, training_labels, "lstm_aspect_level_glove_3", binary_labels=False)
train_deep_lstm_model(embedding_layer, training_text, training_labels, "deep_lstm_aspect_level_glove_3", binary_labels=False)

embedding_layer = embeddings.aspect_level_word2vec_embedding.create_embedding_layer()
training_text, training_labels = embeddings.aspect_level_word2vec_embedding.create_training_corpus(binary_labels=False)
train_cnn_model(embedding_layer, training_text, training_labels, "cnn_aspect_level_word2vec_3", binary_labels=False)
train_lstm_model(embedding_layer, training_text, training_labels, "lstm_aspect_level_word2vec_3", binary_labels=False)
train_deep_lstm_model(embedding_layer, training_text, training_labels, "deep_lstm_aspect_level_word2vec_3", binary_labels=False)

embedding_layer = embeddings.sentence_level_fast_text_embedding.create_embedding_layer()
training_text, training_labels = embeddings.sentence_level_fast_text_embedding.create_training_corpus(binary_labels=False)
train_cnn_model(embedding_layer, training_text, training_labels, "cnn_sentence_level_fast_text_3", binary_labels=False)
train_lstm_model(embedding_layer, training_text, training_labels, "lstm_sentence_level_fast_text_3", binary_labels=False)
train_deep_lstm_model(embedding_layer, training_text, training_labels, "deep_lstm_sentence_level_fast_text_3", binary_labels=False)

embedding_layer = embeddings.sentence_level_glove_embedding.create_embedding_layer()
training_text, training_labels = embeddings.sentence_level_glove_embedding.create_training_corpus(binary_labels=False)
train_cnn_model(embedding_layer, training_text, training_labels, "cnn_sentence_level_fast_text_3", binary_labels=False)
train_lstm_model(embedding_layer, training_text, training_labels, "lstm_sentence_level_fast_text_3", binary_labels=False)
train_deep_lstm_model(embedding_layer, training_text, training_labels, "deep_lstm_sentence_level_fast_text_3", binary_labels=False)

embedding_layer = embeddings.sentence_level_word2vec_embedding.create_embedding_layer()
training_text, training_labels = embeddings.sentence_level_word2vec_embedding.create_training_corpus(binary_labels=False)
train_cnn_model(embedding_layer, training_text, training_labels, "cnn_sentence_level_fast_text_3", binary_labels=False)
train_lstm_model(embedding_layer, training_text, training_labels, "lstm_sentence_level_fast_text_3", binary_labels=False)
train_deep_lstm_model(embedding_layer, training_text, training_labels, "deep_lstm_sentence_level_fast_text_3", binary_labels=False)