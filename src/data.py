import pandas as pd
from os.path import basename, splitext
import glob
import os 
import sys

def construct_training_dataframe(directory='C:/Users/Jerome/OneDrive - Agilytic/Agilytic/Training/nlp_author_case/data/0_raw/training/'):
    res = pd.DataFrame()
    for file_ in glob.glob("{}doc_*.txt".format(directory)):
        with open(file_, encoding='utf-8') as f:
            textf = f.read()
        filename = splitext(basename(file_))[0]
        print(filename)
        res = pd.concat([res,pd.DataFrame(data = {"id" : [filename[6:-3]], 
                                                  "text" : [textf], 
                                                  "author" : [filename[-3:]]})])
    res.to_csv('./data/1_interim/training.csv',index=False)
    return res


#Test
if __name__ == "__main__":
    test_file = 'C:/Users/Jerome/OneDrive - Agilytic/Agilytic/Training/nlp_author_case/data/0_raw/training/doc_id00001MWS.txt'
    with open('C:/Users/Jerome/OneDrive - Agilytic/Agilytic/Training/nlp_author_case/data/0_raw/training/doc_id00001MWS.txt') as f:
        contents =f.read()
        textf = " ".join(line.strip() for line in f)
        print (contents)
        print(textf)