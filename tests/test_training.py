import unittest
from unittest.mock import patch, mock_open
import pandas as pd
import numpy as np
import torch
import json
import os
import sys
import mlflow

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from training.train import IrisTrainer, IrisClassifier

class TestTraining(unittest.TestCase):
    def setUp(self):
        """Set up test environment."""
        self.config = {
            'model': {'hidden_size': 32, 'num_classes': 3, 'learning_rate': 0.01, 'epochs': 5, 'batch_size': 8},
            'paths': {'train_path': 'data/iris_train_data.csv', 'model_save_path': 'models/iris_classifier.pth', 'scaler_save_path': 'models/scaler.pkl'},
            'mlflow': {'tracking_uri': 'http://127.0.0.1:5000', 'experiment_name': 'test_experiment'}
        }
        self.test_data = pd.DataFrame({
            'sepal length (cm)': [5.1, 4.9],
            'sepal width (cm)': [3.5, 3.0],
            'petal length (cm)': [1.4, 1.4],
            'petal width (cm)': [0.2, 0.2],
            'target': [0, 0]
        })
        with patch('builtins.open', mock_open(read_data = json.dumps(self.config))):
            self.trainer = IrisTrainer(config_path = 'settings.json')
        mlflow.set_tracking_uri('http://127.0.0.1:5000')

    def test_load_config(self):
        """Test loading configuration."""
        with patch('builtins.open', mock_open(read_data = json.dumps(self.config))):
            config = self.trainer.load_config('settings.json')
        self.assertEqual(config, self.config)

    @patch('mlflow.sklearn.log_model')
    @patch('pandas.read_csv')
    @patch('os.makedirs')
    @patch('pickle.dump')
    @patch('builtins.open', new_callable = mock_open)
    def test_load_and_preprocess_training_data(self, mock_open, mock_pickle_dump, mock_makedirs, mock_read_csv, mock_mlflow_log_model):
        """Test loading and preprocessing training data."""
        mock_read_csv.return_value = self.test_data
        X, y, input_example = self.trainer.load_and_preprocess_training_data()
        self.assertEqual(X.shape, (2, 4))
        self.assertEqual(y.shape, (2,))
        self.assertIsInstance(input_example, np.ndarray)
        mock_mlflow_log_model.assert_called_once()

    def test_create_data_loader(self):
        """Test creating DataLoader."""
        X = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
        y = torch.tensor([0, 1])
        data_loader = self.trainer.create_data_loader(X, y)
        self.assertIsInstance(data_loader, torch.utils.data.DataLoader)
        self.assertEqual(data_loader.batch_size, self.config['model']['batch_size'])

    def test_initialize_model_and_optimizer(self):
        """Test model and optimizer initialization."""
        model, optimizer, criterion = self.trainer.initialize_model_and_optimizer(input_size=4)
        self.assertIsInstance(model, IrisClassifier)
        self.assertIsInstance(optimizer, torch.optim.Adam)
        self.assertIsInstance(criterion, torch.nn.CrossEntropyLoss)

if __name__ == '__main__':
    unittest.main()