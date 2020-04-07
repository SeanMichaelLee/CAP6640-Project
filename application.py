from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QMessageBox, QLabel, QPlainTextEdit, QComboBox
import re
from keras.models import model_from_json
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
import os
import glob
from dataset import *
import spacy
import pretrained_models.flair
import pretrained_models.text_blob
import pretrained_models.vader

# Signal for submission button selection
def on_click(line_edit, combo_box, training_text):
    # Retrieve the text from the line edit
    text = []
    text.append(str(line_edit.toPlainText()))

    if len(text[0]) > 0:
        preprocess_text(text)

        # Obtain model prediction
        prediction = get_prediction(combo_box, text, training_text)

        if prediction == 0:
            alert = QMessageBox()
            alert.setText("Text Sentiment is Negative.")
            alert.exec_()

        elif prediction == 1:
            alert = QMessageBox()
            alert.setText("Text Sentiment is Positive.")
            alert.exec_()

        else:
            alert = QMessageBox()
            alert.setText("Invalid Input")
            alert.exec_()
    else:
        alert = QMessageBox()
        alert.setText("Invalid Input")
        alert.exec_()

    # Clear the line edit
    line_edit.clear()

def preprocess_text(text):
    TAG_RE = re.compile(r'<[^>]+>')

    # Remove HTML tags
    text[0] = TAG_RE.sub('', text[0])

    # Remove punctuation and numbers
    text[0] = re.sub('[^a-zA-Z]', ' ', text[0])

    # Remove single characters
    text[0] = re.sub(r"\s+[a-zA-Z]\s+", ' ', text[0])

    # Remove multiple spaces
    text[0] = re.sub(r'\s+', ' ', text[0])

    text[0] = text[0].lower()

    return text

def populate_combo_box(combo_box):
    models = []
    for model in glob.glob("models/*.h5"):
        models.append(model)

    # Add pretrained models
    models.append("Flair")
    models.append("TextBlob")
    models.append("VADER")

    combo_box.addItems(models)

def aspect_level_format(training_text):
    # NOTE: Requires running python -m spacy download en_core_web_sm
    nlp = spacy.load('en_core_web_sm')
    training_text = map(lambda x:x.lower(),training_text)
    sentiment_terms = []
    for text in nlp.pipe(training_text):
            if text.is_parsed:
                sentiment_terms.append(' '.join([token.lemma_ for token in text if (not token.is_stop and not token.is_punct and (token.pos_ == "ADJ" or token.pos_ == "VERB"))]))
            else:
                sentiment_terms.append('')
    return sentiment_terms

def get_prediction(combo_box, text, training_text):
    model = combo_box.currentText()

    if "Flair" in model:
        predictions= pretrained_models.flair.predict(text)[0]
    elif "TextBlob" in model:
        prediction = pretrained_models.flair.predict(text)[0]
    elif "VADER" in model:
        prediction = pretrained_models.flair.predict(text)[0]
    else:
        tokenizer = Tokenizer(num_words=5000)
        if "aspect_level" in model:
            training_text = aspect_level_format(training_text)

        tokenizer = Tokenizer(num_words=5000)
        tokenizer.fit_on_texts(training_text)
        text = tokenizer.texts_to_sequences(text)
        text = pad_sequences(text, padding='post', maxlen=1000)

        loaded_model = load_model(model)
        prediction = loaded_model.predict_classes(text, batch_size=1)[0][0]

    return prediction

# Initialize application
application = QApplication([])
training_text, _, _, _ = create_dataset()

# Create a main window
main_window = QWidget()
layout = QVBoxLayout()

# Create UI elements
application_label = QLabel('US Economy Sentiment Analysis Tool');
combo_box = QComboBox()
populate_combo_box(combo_box)
line_edit = QPlainTextEdit()
line_edit.setPlaceholderText("Enter phrase.")
submit_button = QPushButton('Submit')

# Set signals for UI elements
submit_button.clicked.connect(lambda: on_click(line_edit, combo_box, training_text))

# Add UI elements to the layout
layout.addWidget(application_label)
layout.addWidget(line_edit)
layout.addWidget(combo_box)
layout.addWidget(submit_button)

# Set the main window
main_window.setLayout(layout)
main_window.show()

# Execute application
application.exec_()
