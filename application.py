from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QMessageBox, QLabel, QPlainTextEdit
import re
from keras.models import model_from_json
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
import os

# Signal for submission button selection
def on_click(line_edit):
    # Retrieve the text from the line edit
    text = []
    text.append(str(line_edit.toPlainText()))

    if len(text[0]) > 0:
        text = preprocess_text(text)
        text = format_text(text)

        # Load model
        loaded_model = load_model("models/cnn_model.h5")

        # Obtain model prediction
        prediction = loaded_model.predict_classes(text, batch_size=1)[0][0]

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

    return text

def format_text(text):
    # Create a word-to-index dictionary
    tokenizer = Tokenizer(num_words=10000)
    tokenizer.fit_on_texts(text)
    text = tokenizer.texts_to_sequences(text)
    text = pad_sequences(text, padding='post', maxlen=1000)
    return text

# Initialize application
application = QApplication([])

# Create a main window
main_window = QWidget()
layout = QVBoxLayout()

# Create UI elements
application_label = QLabel('US Economy Sentiment Analysis Tool');
line_edit = QPlainTextEdit()
line_edit.setPlaceholderText("Enter phrase.")
submit_button = QPushButton('Submit')

# Set signals for UI elements
submit_button.clicked.connect(lambda: on_click(line_edit))

# Add UI elements to the layout
layout.addWidget(application_label)
layout.addWidget(line_edit)
layout.addWidget(submit_button)

# Set the main window
main_window.setLayout(layout)
main_window.show()

# Execute application
application.exec_()
