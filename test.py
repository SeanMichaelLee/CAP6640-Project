from dataset import *

import glob
import pandas
import embeddings.aspect_level_fast_text_embedding
import embeddings.aspect_level_glove_embedding
import embeddings.aspect_level_word2vec_embedding
import embeddings.sentence_level_fast_text_embedding
import embeddings.sentence_level_glove_embedding
import embeddings.sentence_level_word2vec_embedding
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from keras.models import model_from_json
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
import numpy as np

import pretrained_models.flair
import pretrained_models.text_blob
import pretrained_models.vader

def format_text(text):
    tokenizer = Tokenizer(num_words=50000)
    tokenizer.fit_on_texts(text)
    text = tokenizer.texts_to_sequences(text)
    text = pad_sequences(text, padding='post', maxlen=1000)
    return text

def evaluate(data, model, testing_text, testing_labels, binary_labels=True):
    loaded_model = load_model(model)
    predictions = loaded_model.predict_classes(testing_text, verbose=0)
    
    if not binary_labels:
        testing_labels = [np.where(r==1)[0][0] for r in testing_labels]

    if binary_labels:
        predictions = predictions[:,0]
    else:
        predictions = predictions[:]

    average_type = 'binary' if binary_labels else 'micro'
    
    accuracy = accuracy_score(testing_labels, predictions)
    precision = precision_score(testing_labels, predictions, average=average_type)
    recall = recall_score(testing_labels, predictions, average=average_type)
    f1 = f1_score(testing_labels, predictions, average=average_type)

    return data.append({'Name': model, 'Test Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1': f1}, ignore_index=True)

models = []
for model in glob.glob("models/*.h5"):
    models.append(model)

data = pandas.DataFrame(columns = ['Name', 'Test Accuracy', 'Precision', 'Recall', 'F1'])

# Perform evaluations for trained models
for model in models:
    print("Evaluating: {}".format(model))
    if "aspect_level_fast_text_3" in model:
        testing_text, testing_labels = embeddings.aspect_level_fast_text_embedding.create_testing_corpus(binary_labels=False, )
        data = evaluate(data, model, testing_text, testing_labels, binary_labels=False)

    elif "aspect_level_glove_3" in model:
        testing_text, testing_labels = embeddings.aspect_level_glove_embedding.create_testing_corpus(binary_labels=False)
        data = evaluate(data, model, testing_text, testing_labels, binary_labels=False)

    elif "aspect_level_word2vec_3" in model:
        testing_text, testing_labels = embeddings.aspect_level_word2vec_embedding.create_testing_corpus(binary_labels=False)
        data = evaluate(data, model, testing_text, testing_labels, binary_labels=False)

    elif "sentence_level_fast_text_3" in model:
        testing_text, testing_labels = embeddings.sentence_level_fast_text_embedding.create_testing_corpus(binary_labels=False)
        data = evaluate(data, model, testing_text, testing_labels, binary_labels=False)

    elif "sentence_level_glove_3" in model:
        testing_text, testing_labels = embeddings.sentence_level_glove_embedding.create_testing_corpus(binary_labels=False)
        data = evaluate(data, model, testing_text, testing_labels, binary_labels=False)

    elif "sentence_level_word2vec_3" in model:
        testing_text, testing_labels = embeddings.sentence_level_word2vec_embedding.create_testing_corpus(binary_labels=False)
        data = evaluate(data, model, testing_text, testing_labels, binary_labels=False)
    
    elif "aspect_level_fast_text" in model:
        testing_text, testing_labels = embeddings.aspect_level_fast_text_embedding.create_testing_corpus()
        data = evaluate(data, model, testing_text, testing_labels)

    elif "aspect_level_glove" in model:
        testing_text, testing_labels = embeddings.aspect_level_glove_embedding.create_testing_corpus()
        data = evaluate(data, model, testing_text, testing_labels)

    elif "aspect_level_word2vec" in model:
        testing_text, testing_labels = embeddings.aspect_level_word2vec_embedding.create_testing_corpus()
        data = evaluate(data, model, testing_text, testing_labels)

    elif "sentence_level_fast_text" in model:
        testing_text, testing_labels = embeddings.sentence_level_fast_text_embedding.create_testing_corpus()
        data = evaluate(data, model, testing_text, testing_labels)

    elif "sentence_level_glove" in model:
        testing_text, testing_labels = embeddings.sentence_level_glove_embedding.create_testing_corpus()
        data = evaluate(data, model, testing_text, testing_labels)

    elif "sentence_level_word2vec" in model:
        testing_text, testing_labels = embeddings.sentence_level_word2vec_embedding.create_testing_corpus()
        data = evaluate(data, model, testing_text, testing_labels)
    else:
        print("Invalid model: {}".format(model))

_, testing_text, _, testing_labels = create_dataset()

# Perform evaluations for the TextBlob pretrained sentiment analysis tool
print("Evaluating: TextBlob")
predictions = pretrained_models.text_blob.predict(testing_text)
accuracy = accuracy_score(testing_labels, predictions)
precision = precision_score(testing_labels, predictions)
recall = recall_score(testing_labels, predictions)
f1 = f1_score(testing_labels, predictions)

data = data.append({'Name': "TextBlob", 'Test Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1': f1}, ignore_index=True)

# Perform evaluations for the Flair pretrained sentiment anaylsis tool
print("Evaluating: Flair")
predictions = pretrained_models.flair.predict(testing_text)
accuracy = accuracy_score(testing_labels, predictions)
precision = precision_score(testing_labels, predictions)
recall = recall_score(testing_labels, predictions)
f1 = f1_score(testing_labels, predictions)

data = data.append({'Name': "Flair", 'Test Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1': f1}, ignore_index=True)

# Perform evaluations for the VADER pretrained sentiment anaylsis tool
print("Evaluating: VADER")
predictions = pretrained_models.vader.predict(testing_text)
accuracy = accuracy_score(testing_labels, predictions)
precision = precision_score(testing_labels, predictions)
recall = recall_score(testing_labels, predictions)
f1 = f1_score(testing_labels, predictions)

data = data.append({'Name': "VADER", 'Test Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1': f1}, ignore_index=True)

print(data.to_string())
