FROM python:3.10

# Initialize settings_name variable with the default value
ARG settings_name=settings.json
ENV CONF_PATH=${settings_name}

# Set working directory
WORKDIR /app

# Install any necessary packages listed in requirements.txt
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the training module (needed for IrisClassifier import)
COPY training/ /app/training/

# Copy data folder to Docker
COPY data/ /app/data

# Copy models folder to Docker
COPY models/ /app/models

# Copy the code itself
COPY inference /app/inference

# Copy settings.json
COPY settings.json /app
COPY ${CONF_PATH} /app

# Create directories for data, models, and results
RUN mkdir -p /app/data /app/models /app/inference_results

# Run inference script
RUN python3 inference/run.py