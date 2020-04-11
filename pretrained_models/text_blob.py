from textblob import TextBlob

def predict(text):
    predictions = []
    for sentence in text:
        sentence = TextBlob(sentence)
        predictions.append(1 if sentence.sentiment.polarity >= 0 else 0)

    return predictions
