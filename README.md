# Heart Disease Prediction

## Table of Contents
1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Features](#features)
4. [Methodology](#methodology)
5. [Results](#results)
6. [Usage](#usage)
7. [Contributing](#contributing)
8. [License](#license)

## Introduction
This project aims to develop a machine learning model that can accurately predict the presence or absence of heart disease in patients. The model is designed to assist healthcare professionals in making more timely and accurate diagnoses, ultimately leading to improved treatment effectiveness and patient outcomes.

## Dataset
The dataset used in this project is the Heart Disease dataset, sourced from the Cleveland Clinic Foundation and available on Kaggle. The dataset contains medical records of 1,025 individuals, with 14 features (including age, gender, blood pressure, cholesterol levels, etc.) and a binary target variable indicating the presence or absence of heart disease.

## Features
The dataset contains the following features:

1. Age
2. Gender
3. Chest pain type (4 values)
4. Resting blood pressure
5. Serum cholesterol in mg/dl
6. Fasting blood sugar > 120 mg/dl
7. Resting electrocardiographic results (values 0,1,2)
8. Maximum heart rate achieved
9. Exercise-induced angina
10. Oldpeak = ST depression induced by exercise relative to rest
11. The slope of the peak exercise ST segment
12. Number of major vessels (0-3) colored by fluoroscopy
13. Thal: 0 = normal; 1 = fixed defect; 2 = reversible defect

## Methodology
The project follows these steps:

1. Data exploration and preprocessing
2. Feature selection using Recursive Feature Elimination (RFE)
3. Model construction using logistic regression
4. Model evaluation using various metrics (AUC score, confusion matrix, classification report, ROC curve)
5. Model testing on the unseen test set
6. Visualization of predicted probabilities for both training and test sets

## Results
The developed logistic regression model achieved an AUC score of 0.88 on the test set, indicating good predictive performance. The model's ability to accurately classify heart disease cases was demonstrated through the confusion matrix and classification report.

## Usage
To use the heart disease prediction model, follow these steps:

1. Clone the repository: `git clone https://github.com/your-username/heart-disease-prediction.git`
2. Install the required dependencies: `pip install -r requirements.txt`
3. Run the Python script: `python heart_disease_prediction.py`

The script will output the model's performance metrics and visualizations.

## Contributing
Contributions to this project are welcome. If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

## License
This project is licensed under the [MIT License](LICENSE).
