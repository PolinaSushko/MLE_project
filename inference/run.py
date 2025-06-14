import pandas as pd
import numpy as np
import logging
import json
import os
import sys

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import torch

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from training.train import IrisClassifier

# Setup logging
logging.basicConfig(level = logging.INFO, format = '%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class IrisInference:
    def __init__(self, config_path = "settings.json"):
        self.config = self.load_config(config_path)
        self.model  = None

    def load_config(self, config_path):
        """Load configuration from JSON file"""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            return config
        
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file {config_path} not found")
        
    def load_trained_model(self):
        """Load the trained model from file"""
        try:
            model_path = self.config['paths']['model_save_path']

            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Trained model file not found: {model_path}")
            
            logger.info(f"Loading trained model from {model_path}")

            # Load the saved model
            checkpoint   = torch.load(model_path, weights_only = False)
            model_config = checkpoint['model_config']

            # Initialize model with the saved configuration
            self.model = IrisClassifier(
                input_size = model_config['input_size'],
                hidden_size = model_config['hidden_size'],
                num_classes = model_config['num_classes']
            )

            # Load the trained weights
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()  # Set to evaluation mode

            logger.info(f"Model loaded successfully!")
            logger.info(f"Model architecture: {model_config}")
            logger.info(f"Training metrics: {checkpoint['metrics']}")
        
        except Exception as e:
            logger.error(f"Failed to load trained model: {e}")
            raise

    def load_inference_data(self):
        """Load test data for inference"""
        try:
            inference_path = self.config['paths']['inference_path']
            inference_df   = pd.read_csv(inference_path)

            # Separate features and target
            X_inference = inference_df.drop('target', axis = 1).values
            y_inference = inference_df['target'].values
            
            # Convert to PyTorch tensors
            X_tensor = torch.tensor(X_inference, dtype = torch.float32)
            y_tensor = torch.tensor(y_inference, dtype = torch.long) 
            
            logger.info(f"Inference data loaded: {X_tensor.shape[0]} samples, {X_tensor.shape[1]} features")
            
            return X_tensor, y_tensor

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

    def evaluate(self, y_true, y_pred, probabilities):
        """Evaluate prediction performance"""
        try:
            accuracy = accuracy_score(y_true, y_pred)
            
            # Classification report
            report = classification_report(y_true, y_pred, output_dict = True)
            
            # Confusion matrix
            conf_matrix = confusion_matrix(y_true, y_pred)
            
            # Confidence statistics
            max_probs = np.max(probabilities, axis = 1)
            mean_confidence = np.mean(max_probs)
            low_confidence_samples = np.sum(max_probs < 0.8)
            
            metrics = {
                'accuracy': float(accuracy),
                'classification_report': report,
                'confusion_matrix': conf_matrix.tolist(),
                'mean_confidence': float(mean_confidence),
                'low_confidence_samples': int(low_confidence_samples),
                'total_samples': len(y_true)
            }
            
            logger.info(f"Evaluation metrics calculated")
            logger.info(f"Accuracy: {accuracy:.4f}")
            logger.info(f"Mean confidence: {mean_confidence:.4f}")
            
            return metrics

        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            raise

    def save_results(self, predictions, probabilities, y_true, metrics):
        """Save inference results to files"""
        try:
            results_dir   = os.path.dirname(self.config['paths']['inference_results'])
            os.makedirs(results_dir, exist_ok=True)
            
            # Create detailed results DataFrame
            results_df = pd.DataFrame({
            'true_label': y_true,
            'predicted_label': predictions,
            'confidence': np.max(probabilities, axis = 1),
            'correct': y_true == predictions
        })
            
            # Add probability columns for each class
            for i in range(probabilities.shape[1]):
                results_df[f'prob_class_{i}'] = probabilities[:, i]
            
            # Save results
            results_path = os.path.join(results_dir, 'inference_results.csv')
            results_df.to_csv(results_path, index=False)
            
            # Save metrics
            metrics_path = os.path.join(results_dir, 'inference_metrics.json')
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent = 2, default = str)
            
            logger.info(f"Results saved to {results_dir}")
            
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
            raise
    
    def print_summary(self, metrics):
        """Print inference summary"""
        print("\n" + "="*50)
        print("INFERENCE RESULTS SUMMARY")
        print("="*50)
        print(f"Total samples processed: {metrics['total_samples']}")
        print(f"Overall accuracy: {metrics['accuracy']:.4f}")
        print(f"Mean prediction confidence: {metrics['mean_confidence']:.4f}")
        print(f"Low confidence samples (< 0.8): {metrics['low_confidence_samples']}")
        
        print("\nPer-class Performance:")
        print("-" * 50)
        # Get class indices from classification report (excluding 'accuracy', 'macro avg', 'weighted avg')
        class_keys = [k for k in metrics['classification_report'].keys() 
                    if k not in ['accuracy', 'macro avg', 'weighted avg']]
        
        for class_idx in sorted(class_keys):
            report = metrics['classification_report'][class_idx]
            print(f"Class {class_idx:2} | Precision: {report['precision']:.3f} | "
                f"Recall: {report['recall']:.3f} | F1-score: {report['f1-score']:.3f}")
        
        print("\nConfusion Matrix:")
        print("-" * 20)
        conf_matrix = np.array(metrics['confusion_matrix'])
        print("Predicted:", " ".join(f"Class{i:2}" for i in range(conf_matrix.shape[1])))
        for i, row in enumerate(conf_matrix):
            print(f"Class {i:2}   {row}")
        print("="*50)

    def inference(self):
        """Complete inference pipeline - load model and run predictions"""
        try:
            logger.info("="*50)
            logger.info("STARTING IRIS INFERENCE PIPELINE")
            logger.info("="*50)
            
            # Step 1: Load the trained model from file
            logger.info("Step 1: Loading trained model...")
            self.load_trained_model()
            
            # Step 2: Load inference data
            logger.info("Step 2: Loading inference data...")
            X, y_true = self.load_inference_data()
            
            # Step 3: Make predictions
            logger.info("Step 3: Making predictions...")
            predictions, probabilities = self.predict(X)
            
            # Step 4: Evaluate results
            logger.info("Step 4: Evaluating predictions...")
            metrics = self.evaluate(y_true.numpy(), predictions, probabilities)
            
            # Step 5: Save results
            logger.info("Step 5: Saving results...")
            self.save_results(predictions, probabilities, y_true.numpy(), metrics)
            
            # Print summary
            self.print_summary(metrics)
            
            logger.info("INFERENCE COMPLETED SUCCESSFULLY!")
            
        except Exception as e:
            logger.error(f"Inference pipeline failed: {e}")
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