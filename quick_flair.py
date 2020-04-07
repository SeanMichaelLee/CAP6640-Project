from flair.models import TextClassifier
from flair.data import Sentence

from dataset import *

import numpy as np

from sklearn.model_selection import train_test_split

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
#training_text, testing_text, training_labels, testing_labels = train_test_split(text_list, labels, test_size=0.10, random_state=42)

classifier = TextClassifier.load('en-sentiment')

total_count = correct_count = 0

for text,label in zip(text_list,labels):
    total_count += 1
    sentence = Sentence(text)
    classifier.predict(sentence)

    if((sentence.labels[0].value == 'POSITIVE' and label == 1) or (sentence.labels[0].value == 'NEGATIVE' and label == 0)):
        correct_count += 1

print('Accuracy is: ', correct_count/total_count)
