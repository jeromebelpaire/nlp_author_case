from data import construct_training_dataframe
from features import predictors, bow_vector, tfidf_vector
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn import metrics
import pandas as pd
import os
import sys

#Set main directory 
# if "__file__" in globals():
#     path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#     print(path)
# else:
#     path = os.path.dirname(globals()['_dh'][0])
#     print(path)
# sys.path.insert(0, path)


#Test
training_df = pd.read_csv('./data/1_interim/training.csv') #Read dataset
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


# Logistic Regression Classifier
classifier = RandomForestClassifier(n_estimators=100)

# Create pipeline using Bag of Words
pipe = Pipeline([('cleaner', predictors()),
                ('vectorizer', tfidf_vector),
                ('classifier', classifier)])
# model generation
pipe.fit(X_train,y_train)
print(X_train[0])

#Predict
predicted = pipe.predict(X_test)

# Model Accuracy
print("Random Forest Accuracy:",metrics.accuracy_score(y_test, predicted))
# print("Logistic Regression Precision:",metrics.precision_score(y_test, predicted))
# print("Logistic Regression Recall:",metrics.recall_score(y_test, predicted))


# random_forest_classifier()
#Construct dataset
# construct_training_dataframe()