import pytest
import pandas as pd
import os

def test_data_process():
    """Test data processing script"""
    assert os.path.exists('data/iris_train_data.csv'), "Training file is not created"
    assert os.path.exists('data/iris_inference_data.csv'), "Inference file is not created"

    train_df     = pd.read_csv('data/iris_train_data.csv')
    inference_df = pd.read_csv('data/iris_inference_data.csv')

    assert len(train_df) > 0, "Training data is empty"
    assert len(inference_df) > 0, "Inference data is empty"