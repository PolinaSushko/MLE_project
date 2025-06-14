import json
import pandas as pd
import logging
import os

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Setup logging
logging.basicConfig(level = logging.INFO, format = '%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config(config_path = 'settings.json'):
    """Load configuration from JSON file"""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file {config_path} not found")
    
def load_clean_data():
    """Load and clean the Iris dataset"""
    try:
        # Load data
        iris_sklearn = load_iris()
        iris_df      = pd.DataFrame(data = iris_sklearn.data, columns = iris_sklearn.feature_names)
        iris_df['target'] = iris_sklearn.target
        logger.info(f"Loaded dataset with shape: {iris_df.shape}")

        # Remove any duplicate rows
        initial_rows = len(iris_df)
        iris_df = iris_df.drop_duplicates()
        removed_duplicates = initial_rows - len(iris_df)
        if removed_duplicates > 0:
            logger.info(f"Removed {removed_duplicates} duplicate rows")

        logger.info(f"Target distribution:\n{iris_df['target'].value_counts()}")

        return iris_df

    except Exception as e:
        raise RuntimeError(f"Failed to download dataset: {str(e)}")
    
def normalize_features(df, numeric_cols):
    """Normalize features using StandardScaler"""
    df_scaled = df.copy()

    standard_scaler = StandardScaler()

    df_scaled[numeric_cols] = standard_scaler.fit_transform(df_scaled[numeric_cols])
    logger.info(f"Columns {numeric_cols} were scaled using StandardScaler")

    return df_scaled

def split_data(df, test_size, numeric_cols, target_col):
    """Split data into training and inference sets"""
    # Define X and y arrays
    X = df.drop(target_col, axis = 1).values
    y = df[target_col].values

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, stratify = y, random_state = 42, shuffle = True)

    # Create DataFrames
    train_df = pd.DataFrame(X_train, columns = numeric_cols)
    train_df[target_col] = y_train
    
    inference_df = pd.DataFrame(X_test, columns = numeric_cols)
    inference_df[target_col] = y_test  
    
    logger.info(f"Training set shape: {train_df.shape}")
    logger.info(f"Inference set shape: {inference_df.shape}")
    
    return train_df, inference_df

def save_data(df, file_path):
    """Save DataFrame to CSV file"""    
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok = True)
        
        df.to_csv(file_path, index = False)
        logger.info(f"Data saved to {file_path}")

    except Exception as e:
        raise RuntimeError(f"Failed to save data to {file_path}: {str(e)}")
    
def main():
    """Main data processing pipeline"""
    logger.info("="*50)
    logger.info("STARTING DATA PROCESSING PIPELINE")
    logger.info("="*50)

    try:
        # Load configuration
        config = load_config()

        # Create data directory
        os.makedirs('data', exist_ok = True)

        # Load and clean data
        df_cleaned = load_clean_data()
        
        # Get colunms
        target_col   = 'target'
        numeric_cols = [col for col in df_cleaned.columns if col != target_col]

        # Scale data
        df_scaled = normalize_features(df_cleaned, numeric_cols)
        
        # Split data
        train_df, inference_df = split_data(
            df = df_scaled, 
            test_size = 1 - config['data']['train_test_split'],
            numeric_cols = numeric_cols, 
            target_col = target_col
        )
        
        # Save processed data
        save_data(train_df, config['paths']['train_path'])
        save_data(inference_df, config['paths']['inference_path'])
        
        logger.info("Data processing pipeline completed successfully")

    except Exception as e:
        logger.error(f"Data processing pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()