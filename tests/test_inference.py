import pytest
import os

def test_inference():
    """Test inference script"""
    assert os.path.exists('inference_results/inference_metrics.json'), "Inference metrics file not created"
    assert os.path.exists('inference_results/inference_results.csv'), "Detailed Iiference results file not created"