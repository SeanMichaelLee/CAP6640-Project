from textblob import TextBlob

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

total_count = correct_count = 0

for text,label in zip(text_list,labels):
    total_count += 1
    sentence = TextBlob(text)
    if((sentence.sentiment.polarity >= 0 and label == 1) or (sentence.sentiment.polarity < 0 and label == 0)):
        correct_count += 1

    print('Accuracy is ', correct_count/total_count)