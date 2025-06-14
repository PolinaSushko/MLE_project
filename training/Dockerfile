FROM python:3.10

# Initialize settings_name variable with the default value
ARG settings_name=settings.json
ENV CONF_PATH=${settings_name}

# Set working directory
WORKDIR /app

# Install any necessary packages listed in requirements.txt
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy data folder to Docker
COPY data/ /app/data

# Copy the code itself
COPY training /app/training

# Copy settings.json
COPY settings.json /app
COPY ${CONF_PATH} /app

# Create necessary directories
RUN mkdir -p /app/data /app/models

# Run training script
RUN python3 training/train.py