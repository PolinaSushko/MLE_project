import pandas as pd
import numpy as np
import logging
import json
import os
import sys
import pickle

import torch

import mlflow

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from training.train import IrisClassifier, ScalerWrapper

# Setup logging
logging.basicConfig(level = logging.INFO, format = '%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class IrisInference:
    def __init__(self, config_path = "settings.json"):
        self.config = self.load_config(config_path)
        self.model  = None
        self.scaler = None
        self.client = None

        # Setup MLflow
        self.setup_mlflow()

        # Load trained model
        self.load_trained_model()

    def load_config(self, config_path):
        """Load configuration from JSON file"""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            return config
        
        except FileNotFoundError:
            logger.warning(f"Configuration file {config_path} not found, using default parameters")
            return {
                'paths': {
                        'train_path'      : 'data/iris_train_data.csv',
                        'inference_path'  : 'data/iris_inference_data.csv',
                        'model_save_path' : 'models/iris_classifier.pth',
                        'scaler_save_path': 'models/scaler.pkl',
                        'inference_results' : 'inference_results/'
                    },
                'mlflow': {
                        'experiment_name': 'iris_classification',
                        'tracking_uri': 'http://127.0.0.1:5000',
                        'model_name': 'iris_classifier'
                    }
            }

    def setup_mlflow(self):
        """Setup MLflow tracking"""
        try:
            # Set tracking URI
            mlflow.set_tracking_uri(self.config['mlflow']['tracking_uri'])
            
            # Create an experiment
            experiment_name = f"{self.config['mlflow']['experiment_name']}_inference"
            mlflow.set_experiment(experiment_name)
            
            logger.info(f"MLflow experiment set to: {experiment_name}")

        except Exception as e:
            logger.error(f"Error setting up MLflow: {e}")
            raise

    def load_trained_model(self):
        """Load the trained model from file"""
        try:
            model_path  = self.config['paths']['model_save_path']
            scaler_path = self.config['paths']['scaler_save_path']

            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Trained model file not found: {model_path}")
            
            if not os.path.exists(scaler_path):
                raise FileNotFoundError(f"Scaler file not found: {scaler_path}")
            
            logger.info(f"Loading trained model from {model_path}")
            logger.info(f"Loading scaler from {scaler_path}")

            # Load the saved model
            checkpoint   = torch.load(model_path, weights_only = False)
            model_config = checkpoint['model_config']

            # Initialize model with the saved configuration
            self.model = IrisClassifier(
                input_size  = model_config['input_size'],
                hidden_size = model_config['hidden_size'],
                num_classes = model_config['num_classes']
            )

            # Load the trained weights
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()  # Set to evaluation mode

            # Load the scaler
            with open(scaler_path, 'rb') as f:
                self.scaler = ScalerWrapper(pickle.load(f))

            logger.info(f"Model and scaler loaded successfully!")
            logger.info(f"Model architecture: {model_config}")
            logger.info(f"Training metrics: {checkpoint['metrics']}")
        
        except Exception as e:
            logger.error(f"Failed to load trained model or scaler: {e}")
            raise

    def load_and_preprocess_inference_data(self):
        """Load test data for inference"""
        try:
            inference_path = self.config['paths']['inference_path']
            inference_df   = pd.read_csv(inference_path)

            # Apply the same preprocessing as during training
            X_inference = inference_df.values
            X_inference_scaled = self.scaler.transform(X_inference)
            
            # Convert to PyTorch tensors
            X_tensor = torch.tensor(X_inference_scaled, dtype = torch.float32)
            
            logger.info(f"Inference data loaded and preprocessed: {X_tensor.shape[0]} samples, {X_tensor.shape[1]} features")
            
            return X_tensor, inference_df.copy()

        except FileNotFoundError:
            logger.error(f"Inference data file not found: {inference_path}")
            raise

        except Exception as e:
            logger.error(f"Error loading inference data: {e}")
            raise

    def predict(self, X):
        """Make predictions using the loaded model"""
        try:
            self.model.eval()
            
            with torch.no_grad():
                outputs = self.model(X)
                _, predicted = torch.max(outputs.data, 1)

                probabilities = torch.softmax(outputs, dim = 1)
                predictions   = predicted.numpy()
                probs         = probabilities.numpy()
                
            logger.info(f"Predictions completed for {len(predictions)} samples")
            
            return predictions, probs
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise

    def save_results(self, predictions, probabilities, original_data):
        """Save inference results to files"""
        try:
            results_dir   = os.path.dirname(self.config['paths']['inference_results'])
            os.makedirs(results_dir, exist_ok = True)
            
            # Create detailed results DataFrame
            results_df = original_data.copy()
            results_df['predicted_label'] = predictions
            results_df['confidence'] = np.max(probabilities, axis = 1)
            
            # Add probability columns for each class
            for i in range(probabilities.shape[1]):
                results_df[f'prob_class_{i}'] = probabilities[:, i]
            
            # Save results
            results_path = os.path.join(results_dir, 'inference_results.csv')
            results_df.to_csv(results_path, index = False)
            
            logger.info(f"Results saved to {results_dir}")

            return results_path
            
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
            raise
    
    def print_summary(self, predictions, probabilities):
        """Print inference summary"""
        print("\n" + "="*50)
        print("INFERENCE RESULTS SUMMARY")
        print("="*50)
        print(f"Total samples processed: {len(predictions)}")
        print(f"Average confidence: {np.mean(np.max(probabilities, axis = 1)):.4f}")
        
        # Simple prediction distribution
        pred_distribution = np.bincount(predictions, minlength = 3)
        print("\nPrediction Distribution:")
        print("-" * 30)
        for i, count in enumerate(pred_distribution):
            print(f"Class {i}: {count} samples")
        
        print("="*50)

    def log_inference_metrics_mlflow(self, predictions, probabilities):
        """Log inference metrics to MLflow"""
        try:
            mlflow.log_metric("inference_samples", len(predictions))
            mlflow.log_metric("avg_confidence", np.mean(np.max(probabilities, axis = 1)))
            mlflow.log_metric("min_confidence", np.min(np.max(probabilities, axis = 1)))
            mlflow.log_metric("max_confidence", np.max(np.max(probabilities, axis = 1)))
            
            # Simple prediction distribution
            pred_distribution = np.bincount(predictions, minlength = 3)
            for i, count in enumerate(pred_distribution):
                mlflow.log_metric(f"predicted_class_{i}_count", count)
                mlflow.log_metric(f"predicted_class_{i}_percentage", count / len(predictions) * 100)
            
            mlflow.set_tag("inference_completed", "true")
            
        except Exception as e:
            logger.warning(f"Failed to log inference metrics to MLflow: {e}")

    def inference(self):
        """Complete inference pipeline - load model and run predictions"""
        try:
            logger.info("="*50)
            logger.info("STARTING IRIS INFERENCE PIPELINE")
            logger.info("="*50)

            with mlflow.start_run(run_name = f"iris_inference_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"):
                # Step 1: Load and preprocess inference data
                logger.info("Step 1: Loading and preprocessing inference data...")
                X, original_data = self.load_and_preprocess_inference_data()
                
                # Step 2: Make predictions
                logger.info("Step 2: Making predictions...")
                predictions, probabilities = self.predict(X)
                
                # Step 3: Save results
                logger.info("Step 3: Saving results...")
                results_path = self.save_results(predictions, probabilities, original_data)

                mlflow.log_artifact(results_path, "inference_results")
                
                # Print summary
                self.print_summary(predictions, probabilities)
                
                logger.info("INFERENCE COMPLETED SUCCESSFULLY!")
                logger.info(f"MLflow run ID: {mlflow.active_run().info.run_id}")

                return mlflow.active_run().info.run_id
            
        except Exception as e:
            logger.error(f"Inference pipeline failed: {e}")
            if mlflow.active_run():
                mlflow.set_tag("status", "failed")
                mlflow.log_param("error", str(e))
            raise

def main():
    """Main function to run inference"""
    try:
        inferencer = IrisInference()
        inferencer.inference()
        
    except Exception as e:
        logger.error(f"Main execution failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    main()