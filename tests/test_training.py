import pytest
import os

def test_training():
    """Test training script"""
    assert os.path.exists('models/iris_classifier.pth'), "Model file not created"