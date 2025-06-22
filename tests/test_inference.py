import unittest
from unittest.mock import patch, mock_open, MagicMock
import pandas as pd
import numpy as np
import torch
import json
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from inference.run import IrisInference, IrisClassifier
from training.train import ScalerWrapper

class TestInference(unittest.TestCase):
    def setUp(self):
        """Set up test environment."""
        self.config = {
            'paths': {
                'inference_path': 'data/iris_inference_data.csv',
                'model_save_path': 'models/iris_classifier.pth',
                'scaler_save_path': 'models/scaler.pkl',
                'inference_results': 'inference_results/'
            }
        }
        self.test_data = pd.DataFrame({
            'sepal length (cm)': [5.1, 4.9],
            'sepal width (cm)': [3.5, 3.0],
            'petal length (cm)': [1.4, 1.4],
            'petal width (cm)': [0.2, 0.2]
        })
        self.inferencer = IrisInference()

    def test_load_config(self):
        """Test loading configuration."""
        with patch('builtins.open', mock_open(read_data = json.dumps(self.config))):
            config = self.inferencer.load_config('settings.json')
        self.assertEqual(config, self.config)

    @patch('torch.load')
    @patch('pickle.load')
    @patch('builtins.open', new_callable = mock_open)
    def test_load_trained_model(self, mock_open, mock_pickle_load, mock_torch_load):
        """Test loading trained model and scaler."""
        mock_state_dict = {
        'fc1.weight': torch.randn(64, 4),
        'fc1.bias': torch.randn(64),      
        'fc2.weight': torch.randn(32, 64), 
        'fc2.bias': torch.randn(32),       
        'fc3.weight': torch.randn(3, 32), 
        'fc3.bias': torch.randn(3)
    }
        
        mock_torch_load.return_value = {
            'model_state_dict': mock_state_dict,
            'model_config': {'input_size': 4, 'hidden_size': 64, 'num_classes': 3},
            'metrics': {'accuracy': 0.9}
        }

        mock_underlying_scaler = MagicMock()
        mock_underlying_scaler.transform.return_value = np.array([[1.0, 2.0, 3.0, 4.0]])
        mock_pickle_load.return_value = mock_underlying_scaler

        self.inferencer.load_trained_model()
        
        self.assertIsInstance(self.inferencer.model, IrisClassifier)
        self.assertIsInstance(self.inferencer.scaler, ScalerWrapper)

    @patch('pandas.read_csv')
    def test_load_and_preprocess_inference_data(self, mock_read_csv):
        """Test loading and preprocessing inference data."""
        mock_read_csv.return_value = self.test_data
        self.inferencer.scaler = MagicMock(spec = ScalerWrapper)
        self.inferencer.scaler.transform.return_value = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
        X, original_data = self.inferencer.load_and_preprocess_inference_data()
        self.assertIsInstance(X, torch.Tensor)
        self.assertEqual(X.shape, (2, 4))
        self.assertIsInstance(original_data, pd.DataFrame)

    @patch('torch.nn.Module.eval')
    def test_predict(self, mock_eval):
        """Test making predictions."""
        self.inferencer.model = IrisClassifier(input_size = 4)
        X = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype = torch.float32)
        predictions, probabilities = self.inferencer.predict(X)
        self.assertIsInstance(predictions, np.ndarray)
        self.assertIsInstance(probabilities, np.ndarray)
        self.assertEqual(predictions.shape, (1,))
        self.assertEqual(probabilities.shape, (1, 3))

if __name__ == '__main__':
    unittest.main()