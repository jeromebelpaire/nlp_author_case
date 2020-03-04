import spacy 
import string
# from spacy.lang.en import English
# from spacy.lang.en.stop_words import STOP_WORDS
import pandas as pd
import numpy as np
import os 
import sys
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

#Set main directory 
if "__file__" in globals():
    path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print(path)
else:
    path = os.path.dirname(os.path.dirname(globals()['_dh'][0]))
    print(path)
sys.path.insert(0, path)

# nlp = spacy.load('en_core_web_lg')
# English = spacy.load('en')
parser = spacy.load('en_core_web_sm')
stop_words = parser.Defaults.stop_words
punctuations = string.punctuation

# Creating our tokenizer function
def spacy_tokenizer(sentence):
    # Creating our token object, which is used to create documents with linguistic annotations.
    mytokens = parser(sentence)

    # Lemmatizing each token and converting each token into lowercase
    mytokens = [ word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in mytokens ]

    # Removing stop words
    mytokens = [ word for word in mytokens if word not in stop_words and word not in punctuations ]

    # return preprocessed list of tokens
    return mytokens

# Custom transformer using spaCy
class predictors(TransformerMixin):
    def transform(self, X, **transform_params):
        # Cleaning Text
        return [clean_text(text) for text in X]

    def fit(self, X, y=None, **fit_params):
        return self

    def get_params(self, deep=True):
        return {}

# Custom transformer using spaCy
class predictors2(TransformerMixin):
    def transform(self, X, **transform_params):
        # Cleaning Text
        return [len(text) for text in X]

    def fit(self, X, y=None, **fit_params):
        return self

    def get_params(self, deep=True):
        return {}

class Debug(BaseEstimator, TransformerMixin):

    def transform(self, X):
        print(pd.DataFrame(X).head())
        print(X.shape)
        print(type)
        return X

    def fit(self, X, y=None, **fit_params):
        return self

class Converter(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self

    def transform(self, data_frame):
        return data_frame.values.ravel()

class ArrayCaster(BaseEstimator, TransformerMixin):
  def fit(self, x, y=None):
    return self

  def transform(self, data):
    return np.transpose(np.matrix(data))

# Basic function to clean the text
def clean_text(text):
    # Removing spaces and converting text into lowercase
    return text.strip().lower()

bow_vector = CountVectorizer(tokenizer = spacy_tokenizer, ngram_range=(1,3))
tfidf_vector = TfidfVectorizer(tokenizer = spacy_tokenizer)

#test
# training_df = pd.read_csv('./data/1_interim/training.csv')
# training_case = training_df.iloc[0,2]
# doc = nlp(training_case)
