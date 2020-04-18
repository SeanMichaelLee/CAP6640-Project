import pandas
import seaborn
import matplotlib.pyplot as plt
import re
from sklearn.model_selection import train_test_split
import numpy as np

def create_dataset(binary_labels=True):
    # "U.S. economic performance based on news articles data" obtained from:
    # https://www.figure-eight.com/data-for-everyone/
    us_econmic_newspaper_dataset = pandas.read_csv("data/us-economic-newspaper-clean.csv")
    us_econmic_newspaper_dataset.isnull().values.any()
    us_econmic_newspaper_dataset.shape

    # "Economic News Article Tone and Relevance" obtained from:
    # https://www.figure-eight.com/data-for-everyone/
    econmic_news_dataset = pandas.read_csv("data/Full-Economic-News-clean.csv")
    econmic_news_dataset.isnull().values.any()
    econmic_news_dataset.shape

    # Combine datasets
    dataset = pandas.concat([us_econmic_newspaper_dataset, econmic_news_dataset])
    preprocess_text(dataset)

    # Fill in missing positivity values with 0
    dataset['positivity'] = dataset['positivity'].fillna(0)

    # Treat positivity values in the range of (0.0, 4.0) as negative
    dataset['positivity'] = dataset['positivity'].replace(0, "Neutral")
    dataset['positivity'] = dataset['positivity'].replace(1, "Negative")
    dataset['positivity'] = dataset['positivity'].replace(2, "Negative")
    dataset['positivity'] = dataset['positivity'].replace(3, "Negative")
    dataset['positivity'] = dataset['positivity'].replace(4, "Negative")

    # Treat positivity values in the range of (5.0, 9.0) as positive
    dataset['positivity'] = dataset['positivity'].replace(5, "Positive" if binary_labels else "Neutral")
    dataset['positivity'] = dataset['positivity'].replace(6, "Positive")
    dataset['positivity'] = dataset['positivity'].replace(7, "Positive")
    dataset['positivity'] = dataset['positivity'].replace(8, "Positive")
    dataset['positivity'] = dataset['positivity'].replace(9, "Positive")

    dataset = preprocess_text(dataset)

    # Create the text list from the dataset
    text_list = []
    for index, row in dataset.iterrows():
        text_list.append(row['text'])

    # Create a list of labels from the dataset
    # NOTE: Positive = 2
    #       Neutral = 1
    #       Negative = 0

    labels = []

    if binary_labels:
        labels = np.array(list(map(lambda x: 1 if x=="Positive" else 0, dataset['positivity'])))
    else:
        labels = np.array(list(map(lambda x: [1,0,0] if x=="Positive" else ([0,1,0] if x=="Neutral" else [0,0,1]), dataset['positivity'])))

    # Divide the dataset 10% test and 90% training
    training_text, testing_text, training_labels, testing_labels = train_test_split(text_list, labels, test_size=0.10, random_state=42)
    return training_text, testing_text, training_labels, testing_labels

def run_stats(dataset):
    ax = seaborn.countplot(x='positivity', data=dataset).set_title("Dataset")
    plt.show()

def preprocess_text(dataset):
    TAG_RE = re.compile(r'<[^>]+>')

    for index, row in dataset.iterrows():
        # Remove HTML tags
        row['text'] = TAG_RE.sub('', row['text'])

        # Remove punctuation and numbers
        row['text'] = re.sub('[^a-zA-Z]', ' ', row['text'])

        # Remove single characters
        row['text'] = re.sub(r"\s+[a-zA-Z]\s+", ' ', row['text'])

        # Remove multiple spaces
        row['text'] = re.sub(r'\s+', ' ', row['text'])

        # To Lowercase
        dataset.loc[index, 'text'] = row['text'].lower()

    return dataset
