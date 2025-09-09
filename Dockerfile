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

# Copy the rest of the application code
COPY . .

# Command to run the application
CMD ["conda", "run", "-n", "housing", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
