#  Netflix Recommendation System using Collaborative Filtering
This project implements a movie recommendation system based on user ratings, using the **Surprise** library. The model leverages **collaborative filtering** techniques—specifically **Singular Value Decomposition (SVD)**—to predict unseen user-item interactions and recommend content accordingly.

##  Objective
The aim of this project is to build a scalable and effective recommender system that learns user preferences from historical rating data and suggests movies the user is likely to enjoy. This mimics the recommendation logic used in platforms like Netflix.

## Technologies Used
- Python
- Jupyter Notebook
- Pandas, NumPy
- Matplotlib (for visualization)
- `scikit-surprise` (for collaborative filtering algorithms)

## Project Overview
- Load user-item interaction dataset
- Explore and preprocess the data
- Train a collaborative filtering model using **SVD**
- Evaluate performance using industry-standard metrics (RMSE, MAE)
- Predict ratings and recommend top movies to users

## Model Evaluation
Model performance is validated using **cross-validation**, with the following metrics:
- 🔸 Root Mean Squared Error (**RMSE**)
- 🔸 Mean Absolute Error (**MAE**)
These metrics ensure the accuracy and reliability of the recommendation engine.


