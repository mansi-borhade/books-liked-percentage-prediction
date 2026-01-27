**PREDICTING ACCURACY OF LIKED PERCENTAGE OF BOOKS**

A.PROJECT OVERVIEW

This project focuses on predicting the liked percentage of books using machine learning and deep learning techniques. 
Instead of only recommending books based on ratings or popularity, this system estimates how much a book is likely to be liked by readers, expressed as a liked percentage.
The project is built using:
1.Artificial Neural Network (ANN) for deep learning based regression.
2.Ensemble Learning (Voting Regressor) combining random forest and gradient boosting.
The dataset used is the Goodreads Best Books Ever Dataset, and the work is inspired by and aligned with a PUBLISHED RESEARCH STUDY.

B.PROBLEM STATEMENT

Traditional book recommendation systems rely heavily on ratings, which may not fully capture reader sentiment. This project aims to answer:
1.Can we predict the liked percentage of a book using its metadata such as genre, author, and rating?
By accurately predicting liked percentage, recommendation systems can offer more personalized and sentiment-aware suggestions.

C.SOLUTION APPROACH

The solution follows a two-stage modeling strategy:
1.Artificial Neural Network (ANN): Learns complex, non-linear relationships between book features and liked percentage.
2.Voting Regressor (Ensemble Model): Combines predictions from Random Forest Regressor & Gradient Boosting Regressor by improving accuracy and reduces bias and overfitting.

D.DATASET DESCRIPTION

Source: Goodreads Best Books Ever Dataset
Total Records: ~52,000 books
Features Used:
  ---------------------------------------------------------------------------
| Feature          | Description                                              |
| ---------------- | -------------------------------------------------------- |
| Genre            | Category of the book                                     |
| Author           | Author of the book                                       |
| Rating           | Average Goodreads rating                                 |
| Liked Percentage | Target variable (percentage of users who liked the book) |
 ----------------------------------------------------------------------------
Other columns such as price, pages, language, publisher, etc. are explored but not directly used in final modeling.

E.TECHNOLOGIES AND TOOLS

Programming Language: Python
Libraries: Data Processing: pandas, numpy | Visualization: matplotlib, seaborn | Machine Learning: scikit-learn | Deep Learning: tensorFlow, keras.

F.PROJECT WORKFLOW

1.Data Collection: Dataset loaded from Goodreads open-source data.
2.Data Preprocessing: Removed irrelevant columns, renamed columns for clarity, handled missing values, checked and treated outliers, encoded categorical variables (genre, author) using 
label encoding, standardized numerical features using standardscaler.
3.Exploratory Data Analysis (EDA): Correlation analysis, heatmaps, histograms (ratings, liked percentage, price), genre and language distributions.
[[Key Insight: Rating and liked percentage show strong positive correlation]]
4.Feature Selection: Input Features (X): Encoded genre, encoded author, rating | Target Variable (y): Liked percentage.

**Model Development**
[Artificial Neural Network (ANN)]
*Architecture:
Input Layer: 3 neurons
Hidden Layer 1: 64 neurons (ReLU)
Hidden Layer 2: 32 neurons (ReLU)
Hidden Layer 3: 16 neurons (ReLU)
Output Layer: 1 neuron (Linear)

*Compilation Details
Optimizer: Adam
Loss: Mean Squared Error (MSE)
Epochs: 80
Batch Size: 32

*Performance
MAE: ~1.61
MSE: ~4.02
R² Score: ~0.73 (73%)

[Voting Regressor (Ensemble Model)]

*Base Models
Random Forest Regressor
Gradient Boosting Regressor

*Ensemble Strategy
Soft Voting (Average of predictions)

*Performance
MAE: ~1.42
MSE: ~3.41
R² Score: ~0.76 (76%)

**ENSEMBLE MODEL OUTPERFORMS ANN MODEL**

**Results & Findings**
Books with higher ratings tend to have higher liked percentages.
Fiction and Fantasy genres dominate user preference.
Voting Regressor improves prediction stability and accuracy.
Ensemble learning effectively balances bias and variance.

*RESEARCH REFERENCE*

*This project is based on the research paper:*
*"Predicting Liked Percentage of Book using Artificial Neural Network and Voting Regressor"*
*Published in The Indian Journal of Technical Education, October 2024*
If you find this project useful, please ⭐ the repository....
