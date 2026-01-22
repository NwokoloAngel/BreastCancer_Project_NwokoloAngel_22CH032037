# Breast Cancer Prediction System

This project implements a Machine Learning system to predict whether a breast cancer tumor is Malignant or Benign using the Breast Cancer Wisconsin dataset.

## Features
- **Machine Learning Model**: Support Vector Machine (SVM) trained on the Wisconsin Breast Cancer dataset.
- **Web Interface**: A Flask-based web application to input tumor features and get predictions.
- **Model Persistence**: Uses `joblib` for saving and loading the trained model.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/oekpokpobe2202349-dotcom/BreastCancer_Project_Ekpokpobe-Oghenetejiri-Great_22CH032007.git
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   python app.py
   ```

4. Open your browser and navigate to `http://localhost:5000`.

## Structure
- `model/`: Contains `train_model.py` and the saved `breast_cancer_model.pkl`.
- `templates/`: HTML templates for the web GUI.
- `static/`: Static assets (CSS/JS).
- `app.py`: Flask application entry point.

## Author
**Ekpokpobe Oghenetejiri Great**
Matric Number: 22CH032007
