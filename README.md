
# 📚 Book Liked Percentage Predictor

A machine learning application that predicts the percentage of users who are likely to like a book based on book-level features.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://book-percentage-predictor03.streamlit.app/)
**Live Demo:** https://book-percentage-predictor03.streamlit.app/

## Project Overview

The objective of this project is to predict the liked percentage of a book using structured book metadata.

The final model uses:

- Genre
- Average rating
- Number of ratings
- BBE score
- Number of pages
- Price

The target variable is the book's liked percentage.

## Machine Learning Approach

The project uses a Voting Regressor with multiple regression models combined to improve prediction robustness.

Categorical features such as genre are encoded using the preprocessing pipeline, while numerical features are processed before being passed to the regression models.

The complete preprocessing and prediction pipeline was saved using Joblib.

## Final Model Performance

Final test-set performance:

- MAE: 1.4895%
- MSE: 5.2490
- RMSE: 2.2911%
- R²: 0.8179

Five-fold cross-validation was also performed to evaluate model stability.

Cross-validation results:

- Mean MAE: 1.50%
- Mean RMSE: 2.21%
- Mean R²: 0.8182

## Deployment

The trained model was serialized as:

`final_book_liked_percentage_model.pkl`

A separate deployment dataset containing the required book information was created as:

`final_book_deployment_data.csv`

The model was then integrated into a Streamlit application.

The application supports:

- Searching for books
- Handling multiple books with similar names
- Browsing books by genre
- Selecting a specific book
- Generating a liked-percentage prediction
- Handling missing numerical values
- Restricting predictions to the valid 0–100% range

## Project Structure

```text
books-liked-percentage-prediction/
│
├── app.py
├── final_book_liked_percentage_model.pkl
├── final_book_deployment_data.csv
├── requirements.txt
├── README.md
└── .gitignore
