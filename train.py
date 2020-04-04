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

def train_cnn_model(embedding_layer, training_text, training_labels, name):
    model = cnn.create_cnn(embedding_layer)
    history = cnn.train_cnn(model, training_text, training_labels)

    # Save model
    model.save("models/{}.h5".format(name))
    print("Saved {} to disk".format(name))

def train_lstm_model(embedding_layer, training_text, training_labels, name):
    model = lstm.create_lstm(embedding_layer)
    history = lstm.train_lstm(model, training_text, training_labels)

    # Save model
    model.save("models/{}.h5".format(name))
    print("Saved {} to disk".format(name))

def train_deep_lstm_model(embedding_layer, training_text, training_labels, name):
    model = deep_lstm.create_lstm(embedding_layer, 1000, 2)
    history = deep_lstm.train_lstm(model, training_text, training_labels)

    # Save model
    model.save("models/{}.h5".format(name))
    print("Saved {} to disk".format(name))

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
