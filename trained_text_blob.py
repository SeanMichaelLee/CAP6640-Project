from textblob.classifiers import NaiveBayesClassifier

from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

import embeddings.aspect_level_word2vec_embedding

import pandas

training_text, training_labels = embeddings.aspect_level_word2vec_embedding.create_training_corpus()

train = []
for text, label in zip(training_text, training_labels):
    train.append(tuple((text, label))

cl = NaiveBayesClassifier(train)

testing_text, testing_labels = embeddings.aspect_level_word2vec_embedding.create_testing_corpus()

predictions = cl.classify(testing_text)

data = pandas.DataFrame(columns = ['Name', 'Test Accuracy', 'Precision', 'Recall', 'F1'])

accuracy = accuracy_score(testing_labels, predictions)
precision = precision_score(testing_labels, predictions)
recall = recall_score(testing_labels, predictions)
f1 = f1_score(testing_labels, predictions)

data = data.append({'Name': "TrainedTextBlob", 'Test Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1': f1}, ignore_index=True)

print(data.to_string())