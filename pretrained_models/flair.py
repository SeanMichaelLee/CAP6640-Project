from flair.models import TextClassifier
from flair.data import Sentence

def predict(text):
    classifier = TextClassifier.load('en-sentiment')
    predictions = []
    for sentence in text:
        sentence = Sentence(sentence)
        classifier.predict(sentence)
        predictions.append(1 if sentence.labels[0].value == 'POSITIVE' else 0)

    return predictions
