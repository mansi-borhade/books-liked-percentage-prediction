# Predicting Liked Percentage of Books Using ANN and Voting Regressor Technique 

Machine Learning and Deep Learning Framework for Predicting the Liked Percentage of Books.

## Overview

This repository contains the implementation of a machine learning and deep learning framework for predicting the liked percentage of books using metadata from the Goodreads Best Books Ever Dataset.

The proposed framework combines an Artificial Neural Network (ANN) for learning complex nonlinear relationships and an Ensemble Voting Regressor consisting of Random Forest Regressor and Gradient Boosting Regressor to improve prediction accuracy and model stability.

The objective of this work is to estimate the percentage of readers who are likely to enjoy a particular book based on its metadata, enabling more personalized and sentiment-aware recommendation systems.

## Model Architecture

The proposed framework consists of:

* Artificial Neural Network (ANN) for nonlinear regression.
* Voting Regressor combining Random Forest Regressor and Gradient Boosting Regressor.
* Feature preprocessing and engineering pipeline.
* Performance evaluation using regression metrics.

## Technologies Used

* Python
* TensorFlow
* Keras
* Scikit-learn
* Pandas
* NumPy
* Matplotlib
* Seaborn

## 📊 Dataset

The experiments are performed using the Goodreads Best Books Ever Dataset.

The dataset contains book metadata collected from Goodreads with the following primary fields:

| Column           | Description                                              |
| ---------------- | -------------------------------------------------------- |
| Genre            | Book category                                            |
| Author           | Author name                                              |
| Rating           | Average Goodreads rating                                 |
| Liked Percentage | Percentage of users who liked the book (Target Variable) |

Additional attributes including pages, language, publisher, and price were explored during analysis but were not included in the final predictive model.

## Data Processing

The following preprocessing steps are applied:

* Removal of irrelevant columns.
* Missing value handling.
* Outlier analysis and treatment.
* Feature renaming and cleaning.
* Label encoding of categorical variables.
* Feature standardization using StandardScaler.
* Train-test dataset splitting.

## Exploratory Data Analysis

The exploratory analysis includes:

* Correlation analysis.
* Heatmaps.
* Distribution plots.
* Genre analysis.
* Language distribution.
* Rating distribution.
* Liked percentage distribution.

The analysis demonstrates a strong positive relationship between Goodreads ratings and liked percentage.

## Model Development

### Artificial Neural Network (ANN)

The ANN architecture consists of:

* Input Layer (3 neurons)
* Hidden Layer 1 (64 neurons, ReLU)
* Hidden Layer 2 (32 neurons, ReLU)
* Hidden Layer 3 (16 neurons, ReLU)
* Output Layer (1 neuron, Linear Activation)

Training configuration:

* Optimizer: Adam
* Loss Function: Mean Squared Error (MSE)
* Epochs: 80
* Batch Size: 32

### Voting Regressor

The ensemble model combines:

* Random Forest Regressor
* Gradient Boosting Regressor

The final prediction is obtained by averaging the predictions of both models to improve prediction accuracy and reduce model variance.

## Training Details

The models were developed using:

* Python environment
* TensorFlow and Keras
* Scikit-learn
* Jupyter Notebook / Google Colab

## Evaluation

The models are evaluated using:

* Mean Absolute Error (MAE)
* Mean Squared Error (MSE)
* R² Score

## Results

### Artificial Neural Network

* MAE: ~1.61
* MSE: ~4.02
* R² Score: ~0.73

### Voting Regressor

* MAE: ~1.42
* MSE: ~3.41
* R² Score: ~0.76

The Voting Regressor outperforms the Artificial Neural Network by producing lower prediction error and improved overall regression performance.

## Repository Structure

The repository contains the following files:

README.md: Project documentation and implementation details.

requirements.txt: Python dependencies required to run the project.

book_liked_percentage_prediction.ipynb: Complete Jupyter Notebook containing data preprocessing, exploratory data analysis, feature engineering, ANN implementation, Voting Regressor implementation, model training, prediction, evaluation, and visualization.

## Project Overview

This project implements an Artificial Neural Network (ANN) and an Ensemble Voting Regressor framework for predicting the liked percentage of books. The framework combines deep learning and ensemble machine learning techniques to accurately estimate reader preference using book metadata.

The implementation includes dataset preprocessing, exploratory data analysis, feature engineering, ANN model development, Voting Regressor training, regression evaluation, and comparative performance analysis.

## Dataset Description

The experiments are performed using the Goodreads Best Books Ever Dataset.

The dataset contains:

* Book metadata.
* Genre information.
* Author information.
* Goodreads rating.
* Liked percentage.

The preprocessing pipeline includes:

* Cleaning missing values.
* Removing unnecessary columns.
* Label encoding categorical variables.
* Standardizing numerical features.
* Preparing datasets for machine learning and deep learning models.

## Training Details

The models were trained using:

* TensorFlow/Keras for ANN.
* Scikit-learn for Voting Regressor.
* Adam Optimizer.
* Mean Squared Error Loss.
* Standardized numerical features.

## Evaluation

Performance is evaluated using:

* Mean Absolute Error (MAE)
* Mean Squared Error (MSE)
* R² Score
* Prediction comparison between ANN and Voting Regressor.

## Results

The repository includes:

* Regression performance comparison.
* Prediction accuracy analysis.
* Correlation analysis.
* Exploratory data visualizations.
* Comparative evaluation between ANN and Voting Regressor.

## How to Run

Clone the repository.

Install the required dependencies.

Open the Jupyter Notebook or Google Colab.

Run all cells sequentially.

## Future Work

* Transformer-based recommendation models.
* Explainable recommendation systems.
* Hybrid recommender systems.
* Integration of user reviews using Natural Language Processing.
* Deep hybrid ensemble architectures.
* Larger book recommendation datasets.

## License

This project is intended for academic and research purposes.
