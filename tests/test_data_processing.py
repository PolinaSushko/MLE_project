import unittest
from unittest.mock import patch, mock_open
import pandas as pd
import json
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from data_process.data_processing import load_config, split_data, save_data

class TestDataProcessor(unittest.TestCase):
    def setUp(self):
        """Set up test environment."""
        self.config = {
            'data': {'train_test_split': 0.8},
            'paths': {
                'train_path': 'data/iris_train_data.csv',
                'inference_path': 'data/iris_inference_data.csv'
            }
        }
        self.test_data = pd.DataFrame({
            'sepal length (cm)': [5.1, 4.9],
            'sepal width (cm)': [3.5, 3.0],
            'petal length (cm)': [1.4, 1.4],
            'petal width (cm)': [0.2, 0.2],
            'target': [0, 0]
        })

    def test_load_config(self):
        """Test loading configuration from JSON file."""
        with patch('builtins.open', mock_open(read_data = json.dumps(self.config))):
            config = load_config('settings.json')
        self.assertEqual(config, self.config)

    def test_load_config_file_not_found(self):
        """Test handling of missing configuration file."""
        with patch('builtins.open', side_effect = FileNotFoundError):
            with self.assertRaises(FileNotFoundError):
                load_config('settings.json')

    def test_split_data(self):
        """Test splitting data into train and inference sets."""
        numeric_cols = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
        train_df, inference_df = split_data(self.test_data, test_size = 0.5, numeric_cols = numeric_cols, target_col = 'target')
        self.assertEqual(len(train_df), 1)
        self.assertEqual(len(inference_df), 1)
        self.assertTrue('target' in train_df.columns)
        self.assertFalse('target' in inference_df.columns)

    @patch('pandas.DataFrame.to_csv')
    @patch('os.makedirs')
    def test_save_data(self, mock_makedirs, mock_to_csv):
        """Test saving DataFrame to CSV."""
        save_data(self.test_data, 'data/test.csv')
        mock_makedirs.assert_called_once_with('data', exist_ok = True)
        mock_to_csv.assert_called_once_with('data/test.csv', index = False)

if __name__ == '__main__':
    unittest.main()