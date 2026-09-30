
# Heart Disease Prediction Using Machine Learning

A comparative machine learning project exploring heart disease classification using clinical attributes. The project evaluates Logistic Regression, Random Forest, and K-Nearest Neighbors (KNN), with an emphasis on data preprocessing, feature selection, model evaluation, and reproducibility.

## Project Overview

The objective is to investigate how different machine learning algorithms classify heart disease using structured clinical data.

The project includes:
- Exploratory data analysis and preprocessing.
- Duplicate detection and removal.
- Feature selection using Recursive Feature Elimination (RFE).
- Comparative evaluation of three classification algorithms.
- Performance analysis using multiple evaluation metrics.

## Dataset

The project uses a publicly available Heart Disease dataset sourced from Kaggle.

- Original dataset: 1,025 records
- Input features: 13 clinical attributes
- Target: Binary heart disease classification
- Unique records after duplicate removal: 302

The original dataset contains duplicate observations. To reduce data leakage, duplicate records are removed before splitting the data into training and testing sets.

## Features

The dataset includes the following attributes:

- Age
- Sex
- Chest pain type (cp)
- Resting blood pressure (trestbps)
- Cholesterol (chol)
- Fasting blood sugar (fbs)
- Resting ECG results (restecg)
- Maximum heart rate (thalach)
- Exercise-induced angina (exang)
- ST depression (oldpeak)
- ST segment slope (slope)
- Number of major vessels (ca)
- Thalassemia-related test category (thal)

The target variable indicates the presence or absence of heart disease.

## Methodology

### 1. Data Preprocessing

The dataset is inspected for duplicates and prepared for model training.

Duplicate records are removed before creating the training and test sets.

### 2. Feature Selection

Recursive Feature Elimination (RFE) is applied to Logistic Regression to identify relevant predictive features.

### 3. Machine Learning Models

Three classification algorithms are evaluated:

- Logistic Regression with RFE
- Random Forest Classifier
- K-Nearest Neighbors (KNN)

### 4. Model Evaluation

Model performance is assessed using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion matrix
- Cross-validation

Data preprocessing and feature selection are fitted using training data to avoid information leakage.

## Results

The following results were obtained from the revised evaluation after removing duplicate records.

| Model | Accuracy | F1-score | ROC-AUC |
|---|---:|---:|---:|
| Logistic Regression + RFE | 82.0% | 83.6% | 87.1% |
| Random Forest | 75.4% | 77.6% | 86.1% |
| KNN | 78.7% | 81.2% | 83.8% |

The revised evaluation uses 241 training records and 61 test records.

These findings demonstrate the importance of addressing duplicate observations when evaluating machine learning models.

Results are specific to this dataset and experimental configuration. They should not be interpreted as evidence of clinical diagnostic performance.

## Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn

## Usage

Clone the repository:

```bash
git clone https://github.com/rahaf-11a/Heart-disease-predict-using-ML.git
cd Heart-disease-predict-using-ML
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Run the revised model comparison:

```bash
python compare_heart_disease_models.py --data heart.csv
```

The script evaluates the models and generates performance metrics and visualizations.

## Project Limitations

- The dataset is relatively small after duplicate removal.
- Results require further validation on independent datasets.
- The project is intended for educational and experimental purposes, not clinical diagnosis.

## Acknowledgments

This project was developed collaboratively as part of an academic machine learning project. It builds upon publicly available heart disease data and existing machine learning approaches.
