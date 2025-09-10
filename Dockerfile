# Use Miniconda as the base image
FROM continuumio/miniconda3

# Set the working directory in the container
WORKDIR /app

# Copy the Conda environment file
COPY conda_environment.yml .

# Create the Conda environment
RUN conda env create -f conda_environment.yml

# Activate the environment and install additional packages
RUN conda run -n housing pip install fastapi uvicorn

# Copy the model and data directories
COPY model ./model
COPY src/data ./data

# Copy the application code files individually
COPY src/main.py .
COPY src/test_api.py .
COPY src/test_models.py .
COPY src/create_model.py .
COPY src/create_improved_model.py .
COPY src/evaluate_model.py .
COPY src/index.html .

# Command to run the application
CMD ["conda", "run", "-n", "housing", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
