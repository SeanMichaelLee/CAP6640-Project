from textblob.classifiers import NaiveBayesClassifier

from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

import pandas

from dataset import *

training_text, testing_text, training_labels, testing_labels = create_dataset()

train = []
for text, label in zip(training_text, training_labels):
    train.append(tuple((text, label)))

cl = NaiveBayesClassifier(train)

predictions = []
for text in testing_text:
    predictions.append(cl.classify(text))

data = pandas.DataFrame(columns = ['Name', 'Test Accuracy', 'Precision', 'Recall', 'F1'])

accuracy = accuracy_score(testing_labels, predictions)
precision = precision_score(testing_labels, predictions)
recall = recall_score(testing_labels, predictions)
f1 = f1_score(testing_labels, predictions)

data = data.append({'Name': "TrainedTextBlob (Three Labels)", 'Test Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1': f1}, ignore_index=True)

print(data.to_string())