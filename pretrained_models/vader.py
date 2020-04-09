from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def predict(text):
    predictions = []
    analyser = SentimentIntensityAnalyzer()
    for sentence in text:
        predictions.append(1 if analyser.polarity_scores(sentence)["compound"] >= 0 else 0)

    return predictions
