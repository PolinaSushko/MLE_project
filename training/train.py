import pandas as pd
import numpy as np
import logging
import json
import os
import pickle

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

import torch
from torch.utils.data import TensorDataset, DataLoader
from torch import nn, optim

import mlflow
import mlflow.pytorch
import mlflow.sklearn
from mlflow.models.signature import infer_signature

# Setup logging
logging.basicConfig(level = logging.INFO, format = '%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class IrisClassifier(nn.Module):
    """Neural Network model for Iris classification"""
    def __init__(self, input_size = 4, hidden_size = 64, num_classes = 3):
        """Initialize the neural network"""
        super(IrisClassifier, self).__init__()

        self.fc1      = nn.Linear(input_size, hidden_size)
        self.relu1    = nn.ReLU()
        self.dropout1 = nn.Dropout(0.2)
        self.fc2      = nn.Linear(hidden_size, hidden_size // 2)
        self.relu2    = nn.ReLU()
        self.dropout2 = nn.Dropout(0.2)
        self.fc3      = nn.Linear(hidden_size // 2, num_classes)
        
    def forward(self, x):
        """Forward pass through the network"""
        x = self.dropout1(self.relu1(self.fc1(x)))
        x = self.dropout2(self.relu2(self.fc2(x)))
        x = self.fc3(x)

        return x

class ScalerWrapper:
    """Wrapper for StandardScaler to add predict method for MLflow compatibility"""
    def __init__(self, scaler):
        self.scaler = scaler
        
    def fit(self, X):
        return self.scaler.fit(X)
    
    def transform(self, X):
        return self.scaler.transform(X)
    
    def fit_transform(self, X):
        return self.scaler.fit_transform(X)
    
    def predict(self, X):
        """Predict method for MLflow compatibility - just transforms the data"""
        return self.scaler.transform(X)
    
    def inverse_transform(self, X):
        return self.scaler.inverse_transform(X)

class IrisTrainer:
    def __init__(self, config_path = "settings.json"):
        self.config = self.load_config(config_path)
        self.scaler = None

    def load_config(self, config_path):
        """Load configuration from JSON file"""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            return config
        
        except FileNotFoundError:
            logger.warning(f"Configuration file {config_path} not found, using default parameters")
            return {
                    'model': {
                        'hidden_size'   : 64,
                        'num_classes'   : 3,
                        'learning_rate' : 0.001,
                        'epochs'        : 100,
                        'batch_size'    : 16
                    },
                    'paths': {
                        'train_path'      : 'data/iris_train_data.csv',
                        'inference_path'  : 'data/iris_inference_data.csv',
                        'model_save_path' : 'models/iris_classifier.pth',
                        'scaler_save_path': 'models/scaler.pkl',
                        'inference_results' : 'inference_results/'
                    },
                    'mlflow': {
                        'experiment_name': 'iris_classification',
                        'tracking_uri': 'http://127.0.0.1:5000'
                    }
                }

    def setup_mlflow(self):
        """Setup MLflow tracking"""
        try:
            # Set tracking URI
            mlflow.set_tracking_uri(self.config['mlflow']['tracking_uri'])
            
            # Create an experiment
            experiment_name = self.config['mlflow']['experiment_name']
            mlflow.set_experiment(experiment_name)
            
            logger.info(f"MLflow experiment set to: {experiment_name}")

        except Exception as e:
            logger.error(f"Error setting up MLflow: {e}")
            raise
    
    def load_and_preprocess_training_data(self):
        """Load and preprocess training data from CSV file"""
        try:
            train_path = self.config['paths']['train_path']
            train_df   = pd.read_csv(train_path)

            # Remove any duplicate rows
            initial_rows = len(train_df)
            train_df = train_df.drop_duplicates()
            removed_duplicates = initial_rows - len(train_df)
            if removed_duplicates > 0:
                logger.info(f"Removed {removed_duplicates} duplicate rows")
                mlflow.log_metric("removed_duplicates", removed_duplicates)

            # Separate features and target
            X_train = train_df.drop('target', axis = 1).values
            y_train = train_df['target'].values

            # Scale data
            base_scaler = StandardScaler()
            self.scaler = ScalerWrapper(base_scaler)
            X_train_scaled = self.scaler.fit_transform(X_train)

            # Save scaler for inference
            scaler_path = self.config['paths']['scaler_save_path']
            os.makedirs(os.path.dirname(scaler_path), exist_ok = True)
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
            logger.info(f"Scaler saved to {scaler_path}")
            mlflow.sklearn.log_model(self.scaler, "scaler")
            
            # Convert to PyTorch tensors
            X_tensor = torch.tensor(X_train_scaled, dtype = torch.float32)
            y_tensor = torch.tensor(y_train, dtype = torch.long) 

            mlflow.log_metric("train_samples", X_tensor.shape[0])
            mlflow.log_metric("num_features", X_tensor.shape[1])

            class_distribution = np.bincount(y_train)
            for i, count in enumerate(class_distribution):
                mlflow.log_metric(f"class_{i}_count", count)
            
            logger.info(f"Training data loaded and preprocessed: {X_tensor.shape[0]} samples, {X_tensor.shape[1]} features")
            logger.info(f"Class distribution: {np.bincount(y_train)}")

            input_example = X_train_scaled[:3].astype(np.float32)
            
            return X_tensor, y_tensor, input_example

        except FileNotFoundError:
            logger.error(f"Training data file not found: {train_path}")
            raise

        except Exception as e:
            logger.error(f"Error loading training data: {e}")
            raise

    def create_data_loader(self, X, y):
        """Create DataLoader for training"""
        try:
            dataset     = TensorDataset(X, y)
            data_loader = DataLoader(dataset, batch_size = self.config['model']['batch_size'], shuffle = True)
            logger.info(f"DataLoader created with batch size: {self.config['model']['batch_size']}")

            return data_loader
            
        except Exception as e:
            logger.error(f"Error creating DataLoader: {e}")
            raise

    def initialize_model_and_optimizer(self, input_size):
        """Initialize model, optimizer, and loss function"""
        try:
            # Initialize model
            model = IrisClassifier(
                input_size = input_size,
                hidden_size = self.config['model']['hidden_size'],
                num_classes = self.config['model']['num_classes']
            )
            
            # Initialize optimizer
            optimizer = optim.Adam(
                model.parameters(),
                lr = self.config['model']['learning_rate']
            )
            
            # Initialize loss function
            criterion = nn.CrossEntropyLoss()

            mlflow.log_param("input_size", input_size)
            mlflow.log_param("hidden_size", self.config['model']['hidden_size'])
            mlflow.log_param("num_classes", self.config['model']['num_classes'])
            mlflow.log_param("learning_rate", self.config['model']['learning_rate'])
            mlflow.log_param("batch_size", self.config['model']['batch_size'])
            mlflow.log_param("dropout_rate", 0.2)
            mlflow.log_param("optimizer", "Adam")
            mlflow.log_param("loss_function", "CrossEntropyLoss")

            total_params = sum(p.numel() for p in model.parameters())
            mlflow.log_param("total_parameters", total_params)
            
            logger.info("Model, optimizer, and loss function initialized")
            logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
            
            return model, optimizer, criterion
            
        except Exception as e:
            logger.error(f"Error initializing model and optimizer: {e}")
            raise

    def train_model(self, model, train_dl, optimizer, criterion, num_epochs):
        """
        Performs the complete training loop for a classification model
        """
        try:
            model.train()

            train_losses     = []
            train_accuracies = []

            logger.info(f"Starting training for {num_epochs} epochs...")
            mlflow.log_param("epochs", num_epochs)

            for epoch in range(num_epochs):
                total_loss      = 0
                all_predictions = []
                all_targets     = []

                for xb, yb in train_dl:
                    # zero gradients before forward pass
                    optimizer.zero_grad()

                    # generate predictions
                    pred = model(xb)

                    # calculate loss
                    loss = criterion(pred, yb)

                    # perform a gradient descent
                    loss.backward()
                    optimizer.step()

                    # Track metrics
                    total_loss += loss.item()
                    _, predicted = torch.max(pred.data, 1)
                    all_predictions.extend(predicted.cpu().numpy())
                    all_targets.extend(yb.cpu().numpy())

                # Calculate epoch metrics
                avg_loss = total_loss / len(train_dl)
                accuracy = accuracy_score(all_targets, all_predictions)
                train_losses.append(avg_loss)
                train_accuracies.append(accuracy)

                mlflow.log_metric("train_loss", avg_loss, step = epoch)
                mlflow.log_metric("train_accuracy", accuracy, step = epoch)

                if (epoch + 1) % 10 == 0 or epoch == 0:
                    logger.info(f'Epoch [{epoch + 1}/{num_epochs}] - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}')

            mlflow.log_metric("final_train_loss", train_losses[-1])
            mlflow.log_metric("final_train_accuracy", train_accuracies[-1])
            
            logger.info("Training completed!")
            logger.info(f"Final training accuracy: {train_accuracies[-1]:.4f}")

            return model, train_losses, train_accuracies
        
        except Exception as e:
            logger.error(f"Error during training: {e}")
            raise

    def evaluate_model(self, model, X, y):
        """Evaluate the trained model"""
        try:
            model.eval()
            
            with torch.no_grad():
                outputs = model(X)
                _, predicted = torch.max(outputs.data, 1)
                
                accuracy = accuracy_score(y.numpy(), predicted.numpy())
                
                # Generate classification report
                report = classification_report(y.numpy(), predicted.numpy(), output_dict = True)
                
            mlflow.log_metric("eval_accuracy", accuracy)
            
            for class_label, metrics in report.items():
                if class_label.isdigit():
                    mlflow.log_metric(f"class_{class_label}_precision", metrics['precision'])
                    mlflow.log_metric(f"class_{class_label}_recall", metrics['recall'])
                    mlflow.log_metric(f"class_{class_label}_f1", metrics['f1-score'])
                elif class_label in ['macro avg', 'weighted avg']:
                    mlflow.log_metric(f"{class_label.replace(' ', '_')}_precision", metrics['precision'])
                    mlflow.log_metric(f"{class_label.replace(' ', '_')}_recall", metrics['recall'])
                    mlflow.log_metric(f"{class_label.replace(' ', '_')}_f1", metrics['f1-score'])
            
            logger.info(f"Model evaluation completed - Accuracy: {accuracy:.4f}")
            
            return {
                'accuracy': accuracy,
                'classification_report': report
            }
            
        except Exception as e:
            logger.error(f"Error during model evaluation: {e}")
            raise

    def save_model(self, model, metrics, input_example):
        """Save the trained model and metric"""
        try:
            # Create models directory if it doesn't exist
            model_path = self.config['paths']['model_save_path']
            os.makedirs(os.path.dirname(model_path), exist_ok = True)
            
            # Save model state dict
            torch.save({
                'model_state_dict': model.state_dict(),
                'model_config': {
                    'input_size' : 4,
                    'hidden_size': self.config['model']['hidden_size'],
                    'num_classes': self.config['model']['num_classes']
                },
                'metrics': metrics
            }, model_path)
            
            model.eval()
            with torch.no_grad():
                input_tensor = torch.tensor(input_example, dtype = torch.float32)
                output_example = model(input_tensor).numpy()
                signature = infer_signature(input_example, output_example)
            
            mlflow.pytorch.log_model(
                pytorch_model = model,
                artifact_path = "model",
                signature = signature,
                input_example = input_example,
                registered_model_name = self.config['mlflow']['model_name']
            )
            
            # Логування артефактів
            mlflow.log_artifact(model_path, "traditional_model")
            
            logger.info(f"Model saved to {model_path}")
            logger.info("Model logged to MLflow")
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise

    def train(self):
        """Complete training pipeline"""
        try:
            logger.info("="*50)
            logger.info("STARTING MODEL TRAINING PIPELINE...")
            logger.info("="*50)

            # Setup MLflow
            self.setup_mlflow()
            
            with mlflow.start_run(run_name = f"iris_training_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"):
                mlflow.set_tag("model_type", "neural_network")
                mlflow.set_tag("dataset", "iris")
                mlflow.set_tag("framework", "pytorch")

                # Load data
                logger.info("Step 1: Loading training data...")
                X, y, input_example = self.load_and_preprocess_training_data()
            
                # Create data loader
                logger.info("Step 2: Creating data loaders...")
                train_loader = self.create_data_loader(X, y)
                
                # Initialize model and optimizer
                logger.info("Step 3: Initializing model and optimizer...")
                model, optimizer, criterion = self.initialize_model_and_optimizer(X.shape[1])
            
                # Train model
                logger.info("Step 4: Training model...")
                trained_model, train_losses, train_accuracies = self.train_model(model, train_loader, optimizer, criterion, self.config['model']['epochs'])
            
                # Evaluate model
                logger.info("Step 5: Evaluating model on training data...")
                metrics = self.evaluate_model(trained_model, X, y)
            
                # Save model
                logger.info("Step 6: Saving model...")
                self.save_model(trained_model, metrics, input_example)

                logger.info("="*50)
                logger.info("TRAINING PIPELINE COMPLETED SUCCESSFULLY!")

                return mlflow.active_run().info.run_id
            
        except Exception as e:
            logger.error(f"Training pipeline failed: {e}")
            if mlflow.active_run():
                mlflow.set_tag("status", "failed")
                mlflow.log_param("error", str(e))
            raise

def main():
    """Main function to run model training"""
    try:
        trainer = IrisTrainer()
        trainer.train()
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return 1
        
    return 0

if __name__ == "__main__":
    main()