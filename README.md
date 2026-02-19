# 🎬 Netflix Movie Predictor

Welcome to the **Netflix Movie Predictor**! This web application uses Collaborative Filtering (SVD) to predict user ratings for movies, similar to how Netflix recommends content.

## ✨ Features

*   **SVD Algorithm**: Uses the robust SVD algorithm from the `scikit-surprise` library.
*   **Dynamic Background**: Visuals inspired by the Netflix interface.
*   **Real Data Support**: Capable of training on the full Netflix Prize dataset.
*   **Instant Predictions**: Input a User ID and Movie to get a predicted rating (1-5 stars).

## 📸 Screenshots

![Dashboard](screenshots/dashboard.png)
*Use the prediction interface to estimate movie ratings.*

## 🚀 Getting Started

### Prerequisites

*   Python 3.7+
*   The Netflix Prize Dataset (optional, dummy data included).

### Installation

1.  **Clone the Repo**:
    ```bash
    git clone https://github.com/YOUR_USERNAME/netflix-predictor.git
    cd netflix-predictor
    ```

2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **(Optional) Add Dataset**:
    To use the real model, place `movie_titles.csv` and `combined_data_1.txt` in the root directory.
    *Without these files, the app runs in Demo Mode with dummy movies.*

4.  **Run the App**:
    ```bash
    python app.py
    ```

5.  **Visit**: `http://127.0.0.1:5004`

## 🧠 How it Works

The app checks for the existence of `combined_data_1.txt`. If found, it reads a subset of the data and trains an SVD model. If not found, it gracefully falls back to a demonstration mode using a small list of popular movies and simulated ratings.

## ☁️ Deployment

Ready for generic cloud platforms using `Procfile` and `gunicorn`.
