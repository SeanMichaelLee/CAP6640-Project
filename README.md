# CAP6640-Project
The economy as a whole is very difficult to predict. While many news outlets exist that provide their own views and expectations on where the markets may head, there are always known and unknown factors that end up pushing the market towards its inevitable highs and lows. The ability to predict where the market is headed could provide real-world, monetary benefits to investors as well as a means to foresee declines in the market that may not be obvious from a mere human read of financial news media.

Financial news outlets constantly publish articles that give their own opinions on current events. It is possible to extract sentiment from these publishings. We explore the sentiment analysis of news articles relevant to the United States economy and how that sentiment is related to the actual overall health of the economy. This project contains code for training and testing a variety of trained and pre-trained models with different embeddings:
* Aspect-level CNN (with FastText, Glove, word2vec embeddings)
* Aspect-level LSTM (with FastText, Glove, word2vec embeddings)
* Aspect-level deep LSTM (with FastText, Glove, word2vec embeddings)
* Sentence-level CNN (with FastText, Glove, word2vec embeddings)
* Sentence-level LSTM (with FastText, Glove, word2vec embeddings)
* Sentence-level deep LSTM (with FastText, Glove, word2vec embeddings)
* Flair pre-trained model
* TextBlob pre-trained model
* VADER pre-trained model

## Table Of Contents
- <a href='#installation'>Installation</a>
- <a href='#datasets'>Datasets</a>
- <a href='#training-models'>Training Models</a>
- <a href='#evaluating-models'>Evaluating Models</a>
- <a href='#application'>Application</a>
- <a href='#references'>References</a>

## Installation
This project was created using Python 3.7 and Anaconda 2019.10 the latest version of Python 3.7 and Anaconda can be found [here](https://www.anaconda.com/distribution/).

### Dependencies
The dependencies required for this project can be installed using the following commands:
```Shell
# Install pandas
conda install -c anaconda pandas

# Install seaborn
conda install -c anaconda seaborn

# Install matplotlib.pyplot
conda install -c conda-forge matplotlib

# Install re
conda install -c conda-forge regex

# Install sklearn.model_selection
conda install -c anaconda scikit-learn

# Install numpy
conda install -c anaconda numpy

# Install spacy
conda install -c conda-forge spacy
python -m spacy download en_core_web_sm

# Install keras
conda install -c conda-forge keras

# Install tqdm
conda install -c conda-forge tqdm

# Install nltk
conda install -c anaconda nltk

# Install urllib
conda install -c anaconda urllib3

# Install gzip
conda install -c ostrokach gzip

# Install gensim
conda install -c anaconda gensim

# Install glob
conda install -c conda-forge glob2

# Install PyQt5
conda install -c anaconda pyqt
```

### Embeddings
All of the embeddings (except custom word2vec) used are open source and can be found at:
* glove.6B.100d.txt [1] [](https://nlp.stanford.edu/projects/glove/)
* wiki.en.vec [2] [](https://fasttext.cc)

## Dataset
The dataset used is aggregated from two existing datasets, "U.S. economic performance based on news articles" and "Economic News Article Tone and Relevance" created by Figure Eight [3]. "U.S. economic performance based on news articles" contains 5000 rows of data and "Economic News Article Tone and Relevance" contains 8000 rows of data, both include:
* Trusted judgement : The number of people that judged the positivity of the article.
* Positivity : The positivity of the article with respect to the health of the U.S. economy on a scale 1-9 (1 being negative and 9 being positive).
* Relevance : The relevance of the article with respect to the U.S. economy (yes/no).
* Date : The date the economic article was published.
* Headline : The headline from the economic article.
* Text: The text from the body of the economic article.
We merged the two datasets into a single dataset that utilizes the sentence column and the U.S economy positivity rating column.

We are creating models that predict whether news article sentiment is negative or positive. To do this, we have modified our dataset labels by saying that anything with a tone value below 5 is considered negative, a tone value of 5 is considered neutral, and that anything with a tone value 5 and greater is considered positive. We have split the data into three smaller datasets, allocating 80% for training, 10% for validation and 10% for testing.

The project contains the following csv data files:
* Full-Economic-News-clean
* Full-Economic-News-DFE-839861
* us-economic-newspaper-clean
* us-economic-newspaper
The clean versions of the file contain preprocessed versions of the csv files to remove ambiguous data.

## Training Models
Training the models requires the dependencies, the embedding files, and the csv data files described above. To train all the models with each embedding the following command can be executed:
```Shell
python train.py
```
This will train each model with each embedding for 10 epochs. The trained models will be saved to the "models" directory.

## Evaluating Models
Evaluating the models requires the dependencies, the models, and the csv data files described above. To evaluate the models the following command can be executed:
```Shell
python test.py
```
This will print the evaluation for each model and pre-trained model. The following metrics will be evaluated:
* Accuracy
* Precision
* Recall
* F1

## Application
The project also contains a frontend application, where a user can type a phrase and select a model to perform sentiment analysis. To start the application the following command can be executed:
```Shell
python application.py
```
This will start the application, which the user can interact with.

## References
* [1] Jeffrey Pennington, Richard Socher, and Christopher D. Manning. 2014. GloVe: Global Vectors for Word Representation.
* [2] P. Bojanowski*, E. Grave*, A. Joulin, T. Mikolov, Enriching Word Vectors with Subword Information
* [3] "Data for everyone." [Online]. Available: https://www.figure-eight.com/data-for-everyone/
* [4] Malik, Usman. “Python for NLP: Movie Sentiment Analysis Using Deep Learning in Keras.” Stack Abuse, Stack Abuse, stackabuse.com/python-for-nlp-movie-sentiment-analysis-using-deep-learning-in-keras/.
* [5] Vader:  A  parsimonious  rule-based  model  for  sentiment  analysis  ofsocial  media  text.”  [Online].  Available:  http://comp.social.gatech.edu/papers/icwsm14.vader.hutto.pdf
* [6] “Tutorial:Quickstart.”[Online].Available:https://textblob.readthedocs.io/en/dev/quickstart.html#sentiment-analysis
* [7] A.  Akbik,  D.  Blythe,  and  R.  Vollgraf,  “Contextual  string  embeddingsfor sequence labeling,” inCOLING 2018, 27th International Conferenceon Computational Linguistics, 2018, pp. 1638–1649.
