## ⚽ La Liga Match Predictor

A production-ready end-to-end Machine Learning application that predicts Spanish La Liga match outcomes. This project demonstrates a complete MLOps lifecycle, from modular data engineering to automated cloud deployment.

🌐 **[View Live Web App](http://13.40.77.75:8501/)**


# 🎯 Project Overview

This repository implements a robust ML pipeline that transforms historical football data into actionable match insights. It focuses on clean code architecture, scalability, and automated delivery.

# ⚙️ End-to-End ML Workflow

This project follows a professional lifecycle to ensure reproducibility and production readiness:

Data Ingestion: Automatically collects raw match data from source directories and splits it into training and testing sets while maintaining data integrity.

Data Transformation: Implements feature engineering and preprocessing pipelines (scaling, encoding) to convert raw data into model-ready features.

Model Training: Utilizes a Random Forest Classifier with hyperparameters tuned via Bayesian Search (Scikit-optimize) to maximize predictive accuracy.

Model Evaluation: Validates performance using industry-standard metrics (Accuracy, Precision, Recall) to ensure the model outperforms baseline benchmarks.

MLOps & Logging: Every step is tracked via a custom logging module, with a centralized exception handling system for robust debugging.

Deployment (CI/CD): Uses GitHub Actions to automatically trigger deployments to an AWS EC2 instance, ensuring the live app is always in sync with the latest code.

# ✨ Key Features

Match Predictions: Instant Win/Draw/Away Win probabilities for 26 La Liga teams.

Interactive UI: Clean, responsive Streamlit dashboard with visual analytics.

Production Standards: Decoupled architecture (Ingestion → Transformation → Training).

Automated CI/CD: Real-time deployment to AWS on every code push.

# 🛠️ Tech Stack

Core: Python 3.13, Scikit-learn, Pandas, NumPy

Optimization: Scikit-optimize (Bayesian Search)

Frontend: Streamlit

DevOps: GitHub Actions, AWS EC2, Systemd



# 🚀 Quick Start

1. Clone & Setup

git clone [https://github.com/TanishS2003/end-to-end_ml_project.git](https://github.com/TanishS2003/end-to-end_ml_project.git)

cd end-to-end_ml_project

pip install -r requirements.txt


2. Train & Run

# Retrain the model and generate artifacts
python train_pipeline.py

# Launch the web app
streamlit run app.py
