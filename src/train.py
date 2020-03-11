from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import make_column_transformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
import pandas as pd
import os
import sys
import random
import pickle

from src.data import construct_training_dataframe
from src.config import Config
from src.features import predictors, bow_vector, tfidf_vector, predictors2

def train(model_name, training_csv_path=''):
    '''
    Train model based on model name:
    Available modelling pipelines:
        'logistic_regression_classifier'
        'random_forest_classifier'
        'multinomial_classifier'
        'multinomial_classifier_countvectorizer'
        'deep_learning_ulm_fit' -> This one is not implemented here, 
                                   but in Google Colab

    '''
    #Setup
    config = Config()
    seed = random.seed(42)

    #Load training set
    if training_csv_path == '':
        training_df = pd.read_csv('{}training.csv'.format(config.get_interim_data_path())) #Read dataset
        print(training_df.head())

    #Split train test
    X = training_df['text'] # the features we want to analyze
    y = training_df['author'] # the labels, or answers, we want to test against
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed)

    #Apply and save model
    if model_name == 'logistic_regression_classifier':
        logistic_regression_classifier(X_train, X_test, y_train, y_test)
    elif model_name == 'random_forest_classifier':
        random_forest_classifier(X_train, X_test, y_train, y_test)
    elif model_name == 'multinomial_classifier':
        multinomial_classifier(X_train, X_test, y_train, y_test)
    elif model_name == 'multinomial_classifier_countvectorizer':
        multinomial_classifier_countvectorizer(X_train, X_test, y_train, y_test)
    elif model_name == 'deep_learning_ulm_fit':
        deep_learning_ulm_fit(X_train, X_test, y_train, y_test)
    else:
        raise NotImplementedError('This model is not Known')

def logistic_regression_classifier(X_train, X_test, y_train, y_test):
    '''
    Train with countvectorizer and Logistic Regression
    Performance with seed 42: 0.77
    '''

    # Logistic Regression Classifier
    classifier = LogisticRegression()

    # Create pipeline using Bag of Words
    pipe = Pipeline([("cleaner", predictors()),
                    ('vectorizer', bow_vector),
                    ('classifier', classifier)])
    # model generation
    pipe.fit(X_train,y_train)

    #Predict
    predicted = pipe.predict(X_test)

    #Save
    save_path = '{}logistic_regression_classifier.pkl'.format(config.get_models_path())
    with open(save_path,'wb') as f:
        pickle.dump(pipe,f)

    # Model Accuracy
    print("Logistic Regression Accuracy:",metrics.accuracy_score(y_test, predicted))

def random_forest_classifier(X_train, X_test, y_train, y_test):
    '''
    Train with tfidf and Random forest
    Performance with seed 42: 0.70
    '''
    classifier = RandomForestClassifier(n_estimators=100)

    # Create pipeline using Bag of Words
    pipe = Pipeline([('cleaner', predictors()),
                    ('vectorizer', tfidf_vector),
                    ('classifier', classifier)])

    # Fit pl to the training data
    pipe.fit(X_train, y_train)

    #Predict
    predicted = pipe.predict(X_test)

    #Save
    save_path = '{}rano.pkl'.format(config.get_models_path())
    with open(save_path,'wb') as f:
        pickle.dump(pipe,f)

    # Model Accuracy
    print("Random Forest Accuracy:",metrics.accuracy_score(y_test, predicted))

def multinomial_classifier(X_train, X_test, y_train, y_test):
    '''
    Train with tfidf vector and Multinomial classifier
    Performance with seed 42: 0.80
    '''
    classifier = MultinomialNB()

    # Create pipeline using Bag of Words
    pipe = Pipeline([('cleaner', predictors()),
                    ('vectorizer', tfidf_vector),
                    ('classifier', classifier)])

    # Fit pl to the training data
    pipe.fit(X_train, y_train)

    #Predict
    predicted = pipe.predict(X_test)

    #Save
    save_path = '{}multinomial_classifier.pkl'.format(config.get_models_path())
    with open(save_path,'wb') as f:
        pickle.dump(pipe,f)

    # Model Accuracy
    print("Multinomial Accuracy:",metrics.accuracy_score(y_test, predicted))

def multinomial_classifier_countvectorizer(X_train, X_test, y_train, y_test):
    '''
    Train with countvectorizer and Multinomial classifier
    Performance with seed 42: 0.84
    '''
    classifier = MultinomialNB()

    # Create pipeline using Bag of Words
    pipe = Pipeline([('cleaner', predictors()),
                    ('vectorizer', bow_vector),
                    ('classifier', classifier)])

    # Fit pl to the training data
    pipe.fit(X_train, y_train)

    #Predict
    predicted = pipe.predict(X_test)

    #Save
    save_path = '{}multinomial_classifier_countvectorizer.pkl'.format(config.get_models_path())
    with open(save_path,'wb') as f:
        pickle.dump(pipe,f)

    # Model Accuracy
    print("Multinomial Accuracy:",metrics.accuracy_score(y_test, predicted))

def deep_learning_ulm_fit(X_train, X_test, y_train, y_test):
    '''
    Train with deeplearning using ULMfit data
    Performance with seed 42: 0.86
    '''
    raise NotImplementedError('This model should be Trained using Google Colab for GPU power, please see ./scripts/train_ulm_fit.ipynb')

if __name__ == "__main__":
    train('deep_learning_ulm_fit')
    # train('logistic_regression_classifier')