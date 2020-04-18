from flair.data import Corpus
from flair.datasets import TREC_6, CSVClassificationCorpus
from flair.embeddings import WordEmbeddings, StackedEmbeddings, FlairEmbeddings, DocumentRNNEmbeddings, CharacterEmbeddings, BertEmbeddings, ELMoEmbeddings, DocumentPoolEmbeddings, DocumentRNNEmbeddings
from flair.models import TextClassifier
from flair.trainers import ModelTrainer
from flair.data import Sentence, Dictionary
from flair.models import LanguageModel
from flair.trainers.language_model_trainer import LanguageModelTrainer
from flair.visual.training_curves import Plotter

from tqdm import tqdm

import torch

from textblob.classifiers import NaiveBayesClassifier

from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

import embeddings.aspect_level_word2vec_embedding

import pandas

import xgboost

dictionary: Dictionary = Dictionary.load('chars')
data_folder = 'data/two-label-corpus'
column_name_map = {0: "label", 1: "text"}

corpus = CSVClassificationCorpus(data_folder, column_name_map)

label_dict = corpus.make_label_dictionary()

word_embeddings = [WordEmbeddings('glove')]

document_embeddings: DocumentRNNEmbeddings = DocumentRNNEmbeddings(word_embeddings, hidden_size=512)

classifier = TextClassifier(document_embeddings, label_dictionary=label_dict)

trainer = ModelTrainer(classifier, corpus)

trainer.train('data/two-label-corpus/train', learning_rate=0.6, mini_batch_size=32, anneal_factor=0.5, patience=5, max_epochs=10)

plotter = Plotter()
plotter.plot_weights('weights.txt')