# Iris Classification Project
A complete Machine Learning Engineering project for Iris flower classification using PyTorch and Docker.
This project implements an end-to-end machine learning pipeline for classifying Iris flowers into three species. The project follows MLOps best practices with containerized training and inference pipelines.

## Features
- **Data Processing:** Automated data loading, splitting and normalization
- **Deep Learning:** PyTorch neural network with dropout regularization
- **Containerization:** Separate Docker containers for training and inference
- **Logging:** Detailed logging throughout the pipeline
- **Configuration:** JSON based configureation management

## Project structure
```
MLE_PROJECT/
├── data/                                # Data files used for training and inference (it can be generated with data_generation.py script)
│   ├── iris_train_data.csv
│   └── iris_inference_data.csv
├── data_process/                        # Scripts used for data processing
│   ├── __init__.py
│   └── data_preparation.py
├── training/                            # Scripts and Dockerfiles used for training
│   ├── __init__.py
│   ├── Dockerfile
│   └── train.py
├── inference/                           # Scripts and Dockerfiles used for inference
│   ├── __init__.py
│   ├── Dockerfile
│   └── run.py
├── inference_results/                   # Inference results
│   ├── inference_metrics.json
│   └── inference_results.csv
├── models/                              # Trained models storage
│   └── (model files will be saved here)
├── tests/                               # Unit tests
│   ├── __init__.py
│   ├── test_data_processing.py
│   ├── test_training.py
│   └── test_inference.py
├── settings.json                        # All configurable parameters and settings
├── .gitignore
├── README.md
├── utils/ # Utility functions
└── requirements.txt                     # Base requirements
```

## Quick start
**1. Install Docker:** Ensure Docker is installed on your system.

**2. Clone the Repository:**
```
git clone https://github.com/PolinaSushko/MLE_project.git
cd MLE_project
```
**3. Prepare Data:**
```
python data_process/data_processing.py
```
**4. Build and Run Training:**
```
# Build training image
docker build -t iris-training -f ./training/Dockerfile .

# Run training
docker run -it iris-training python training/train.py /bin/bash
```
Then, move the trained model from the directory inside the Docker container /app/models to the local machine using:
```
# List all containers
docker ps -a

# Copy from the stopped container 
# Replace <container_id> with your Docker container ID
docker cp <container_id>:/app/models/iris_classifier.pth ./models/iris_classifier.pth
```
Alternatively, the train.py script can also be run locally as follows:
```
python training/train.py
```
**5. Build and Run Inference:**
```
# Build inference image
docker build -t iris-inference -f ./inference/Dockerfile .

# Run inference
docker run -it iris-inference python inference/run.py /bin/bash
```
Alternatively, you can also run the inference script locally as follows:
```
python inference/run.py
```

## Testing
Run unit tests:
```
pytest tests/ 
```

## Configuration
Modify `settings.json` to adjust:
- Model hyperparameters
- Data split ratios
- File paths
- Trining epochs

## Results
After inference, check the `inference_results/` directory for:
- `inference_results.csv`: Detailed predictions
- `inference_metrics.json`: Performance metrics

## Requirements
- Python 3.9+
- Docker
- PyTorch
- Pandas
- Scikit-learn
- Numpy
- Pytest