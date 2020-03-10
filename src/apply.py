from config import Config
import pandas as pd
import os
import sys
import pickle
from joblib import dump, load
# import fastai
from fastai.text import load_learner, DatasetType

config = Config()

def apply_multinomial_classifier_countvectorizer(input_path):
    '''
    Apply saved multinomial classifier on csv file with one or multiple lines
    '''
    # load
    save_path = '{}multinomial_classifier_countvectorizer.pkl'.format(config.get_models_path())
    with open(save_path, 'rb') as f:
        model = pickle.load(f)

    input_df = pd.read_csv(input_path) #Read dataset
    # input_df = input_df.iloc[0:100] #Truncate rows
    print(input_df.head())

    prob_preds = model.predict_proba(input_df['text'])

    result_df = pd.DataFrame({'id': input_df['id'],
                       'EAP':prob_preds[:,0],
                       'HPL':prob_preds[:,1],
                       'MWS':prob_preds[:,2]})

    result_df['prediction'] = result_df.iloc[:,1:].idxmax(axis=1)

    result_df.to_csv('{}results.csv'.format(config.get_processed_data_path()), index=False)  

    return result_df

def apply_fastai_model(input_path):
    '''
    Apply saved fastai (ULM_fit) classifier on csv file with one or multiple lines
    Carefull: slow without GPU
    '''
     # load
    learn = load_learner(config.get_models_path())

    input_df = pd.read_csv(input_path)
    # input_df = input_df.iloc[0:3] #Truncate rows
    learn.data.add_test(input_df['text'])
    print(input_df.head())

    prob_preds = learn.get_preds(ds_type=DatasetType.Test, ordered=True)
    
    result_df = pd.DataFrame({'id': input_df['id'],
                              'EAP':prob_preds[0].numpy()[:,0],
                              'HPL':prob_preds[0].numpy()[:,1],
                              'MWS':prob_preds[0].numpy()[:,2]})

    result_df['prediction'] = result_df.iloc[:,1:].idxmax(axis=1)

    result_df.to_csv('{}results_fastai.csv'.format(config.get_processed_data_path()), index=False)  

    return result_df

def apply_fastai_model_on_sentence(sentence):
    '''
    Apply saved fastai (ULM_fit) classifier on sentence
    '''
     # load
    learn = load_learner(config.get_models_path())

    # Predict
    result = learn.predict(sentence)

    return str(result[0])

if __name__ == "__main__":
    #Test case
    config = Config()
    result = apply_fastai_model_on_sentence('Idris was very content')
    print(result)