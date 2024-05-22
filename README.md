# Custom Linear Regression Model

## Overview
This repository contains all the necessary code and datasets to train and evaluate a custom linear regression model, as outlined in the Machine Learning Homework 1. This model includes implementations for batch, mini-batch, and stochastic gradient descent, as well as L1 and L2 regularization.

## Features
- Custom linear regression implementation.
- Support for polynomial regression.
- Regularization techniques: L1 (Lasso) and L2 (Ridge).
- Evaluation metrics: Mean Squared Error (MSE) and R-squared.

## Dataset
The `insurance.csv` dataset used in this project includes various features such as age, sex, BMI, children, smoker status, region, and insurance charges. The dataset is split into training and test sets to evaluate the model's performance.

Additionally, the `rideshare_kaggle.csv` dataset is utilized for analyzing ride-sharing data. You can download this dataset from [this link](https://www.kaggle.com/datasets/mohan28169/rideshare-kagglecsv).

## Prerequisites
Before you begin, ensure you have met the following requirements:
- Python 3.6 or higher
- Libraries: NumPy, pandas, matplotlib, scikit-learn

You can install the required packages using the following command:
```bash
pip install numpy pandas matplotlib scikit-learn
