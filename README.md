# House Price Prediction API

This project implements a machine learning model for house price prediction as a scalable RESTful service.

## Project Structure

- `docs/` - Documentation files (presentations in Markdown and PDF formats)
- `src/` - Source code files (Python scripts, Dockerfile, HTML interface, data)
- `model/` - Trained model artifacts

## Quick Start

1. Install dependencies using Conda:
   ```bash
   conda env create -f src/conda_environment.yml
   conda activate housing
   ```

2. Train the model:
   ```bash
   python src/create_model.py
   python src/create_improved_model.py
   ```

3. Evaluate model performance:
   ```bash
   python src/evaluate_model.py
   ```

4. Run the API service with Docker:
   ```bash
   docker build -t house-price-prediction -f src/Dockerfile .
   docker run -p 8000:8000 house-price-prediction
   ```

5. Test the API:
   ```bash
   python src/test_api.py
   ```

## API Usage

After starting the service, you can access the API at `http://localhost:8000/predict` by sending a POST request with JSON data containing house features.

For detailed technical information about the implementation, see `docs/README.md`.