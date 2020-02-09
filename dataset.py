import pandas
import seaborn
import matplotlib.pyplot as plt
import re

def create_dataset():
    # "U.S. economic performance based on news articles data" obtained from:
    # https://www.figure-eight.com/data-for-everyone/
    us_econmic_newspaper_dataset = pandas.read_csv("data/us-economic-newspaper.csv")
    us_econmic_newspaper_dataset.isnull().values.any()
    us_econmic_newspaper_dataset.shape

    # "Economic News Article Tone and Relevance" obtained from:
    # https://www.figure-eight.com/data-for-everyone/
    econmic_news_dataset = pandas.read_csv("data/Full-Economic-News-DFE-839861.csv")
    econmic_news_dataset.isnull().values.any()
    econmic_news_dataset.shape

    # Combine datasets
    dataset = pandas.concat([us_econmic_newspaper_dataset, econmic_news_dataset])

    # Fill in missing positivity values with 0
    dataset['positivity'] = dataset['positivity'].fillna(0.0)

    # Treat positivity values of 0 as neutral
    dataset['positivity'] = dataset['positivity'].replace(0.0, "Neutral")

    # Treat positivity values in the range of (1.0, 4.0) as negative
    dataset['positivity'] = dataset['positivity'].replace(1.0, "Negative")
    dataset['positivity'] = dataset['positivity'].replace(2.0, "Negative")
    dataset['positivity'] = dataset['positivity'].replace(3.0, "Negative")
    dataset['positivity'] = dataset['positivity'].replace(4.0, "Negative")

    # Treat positivity values in the range of (5.0, 9.0) as positive
    dataset['positivity'] = dataset['positivity'].replace(5.0, "Positive")
    dataset['positivity'] = dataset['positivity'].replace(6.0, "Positive")
    dataset['positivity'] = dataset['positivity'].replace(7.0, "Positive")
    dataset['positivity'] = dataset['positivity'].replace(8.0, "Positive")
    dataset['positivity'] = dataset['positivity'].replace(9.0, "Positive")

    return dataset

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

    return dataset
