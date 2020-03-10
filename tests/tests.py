import os
import sys
import pytest

proj_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(proj_dir)

from src.apply import *

def test_apply_fastai_model_on_sentence():
    expected = len('MWS')
    assert len(apply_fastai_model_on_sentence('Idris was very content')) == expected