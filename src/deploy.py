from config import Config
from fastai.text import load_learner
import torch
import tarfile
import numpy as np
import pandas as pd

#Configuration
config = Config()
#load
learn = load_learner(config.get_models_path())
# export model to TorchScript format
path_img = config.get_models_path()
# trace_input = torch.ones(400,14352,1152,3).cpu()
# trace_input = torch.ones(1,3,400).cpu()
# trace_input = torch.ones(1,5,400,1152,3).cpu()
# trace_input = torch.randint(0, 400, (1152,5)).cpu()
df = pd.DataFrame({'text':['This is it']})
trace_input = df['text']
# trace_input = torch.randint(0, 100, (10,5)).cpu()

jit_model = torch.jit.trace(learn.model.float(), trace_input)
model_file='nlp_author.pth'
output_path = str(path_img/f'models/{model_file}')
torch.jit.save(jit_model, output_path)
# export classes text file
# save_texts(path_img/'{}classes.txt'.format(config.get_models_path()), data.classes)
tar_file=path_img/'{}model.tar.gz'.format(config.get_models_path())
classes_file='classes.txt'
# create a tarfile with the exported model and classes text file
with tarfile.open(tar_file, 'w:gz') as f:
    f.add(path_img/f'models/{model_file}', arcname=model_file)
    f.add(path_img/f'models/{classes_file}', arcname=classes_file)