from data import construct_training_dataframe
from config import Config
from features import predictors, bow_vector, tfidf_vector, predictors2, Debug, Converter, ArrayCaster
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

#Set main directory 
# if "__file__" in globals():
#     path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#     print(path)
# else:
#     path = os.path.dirname(globals()['_dh'][0])
#     print(path)
# sys.path.insert(0, path)

config = Config()
random.seed(42)

#Test
training_df = pd.read_csv('{}training.csv'.format(config.get_interim_data_path())) #Read dataset
print(training_df.head())

#Split train test
X = training_df['text'] # the features we want to analyze
y = training_df['author'] # the labels, or answers, we want to test against
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

def logistic_regression_classifier():
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

    # Model Accuracy
    print("Logistic Regression Accuracy:",metrics.accuracy_score(y_test, predicted))
    # print("Logistic Regression Precision:",metrics.precision_score(y_test, predicted))
    # print("Logistic Regression Recall:",metrics.recall_score(y_test, predicted))
    return predicted

def random_forest_classifier():
    '''
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

    # Model Accuracy
    print("Random Forest Accuracy:",metrics.accuracy_score(y_test, predicted))

def multinomial_classifier():
    '''
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

    # Model Accuracy
    print("Multinomial Accuracy:",metrics.accuracy_score(y_test, predicted))

def nested_pipelines():
    # Create a FeatureUnion with nested pipeline: process_and_join_features
    process_and_join_features = FeatureUnion(
                transformer_list = [
                    ('feat1', Pipeline([
                        ('cleaner', predictors()),
                        ('vectorizer', tfidf_vector)
                    ])),
                    ('feat2', Pipeline([
                        ('cleaner2', predictors2()),
                        ('caster', ArrayCaster())
                    ]))
                 ]
            )

    # Instantiate nested pipeline: pl
    pipe = Pipeline([
            ('union', process_and_join_features),
            ('classifier', classifier)
        ])


if __name__ == "__main__":
    multinomial_classifier()