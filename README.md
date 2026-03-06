## ⚽ La Liga Match Predictor

A production-grade, end-to-end Machine Learning application that predicts Spanish La Liga outcomes using a high-performance ensemble of Deep Learning and Gradient Boosting. This project demonstrates a complete MLOps lifecycle, from advanced feature engineering to automated cloud deployment.

🌐 **[View Live Web App](http://13.40.77.75:8501/)**

# 🎯 Project Overview

This repository implements a robust ML pipeline that transforms raw football data into actionable match insights. By analyzing underlying "dominance" metrics—like Shots on Target and Corner pressure—this project breaks the 50% accuracy barrier in a three-class problem, providing a more scientific alternative to traditional win/loss modeling.

# ⚙️ Advanced ML Workflow

This project follows a professional lifecycle designed for high-stakes temporal data:

- **Data Ingestion**: Automatically downloads and merges 6 seasons of La Liga data, maintaining strict chronological integrity and sorting.
- **Feature Engineering (Underlying Metrics)**: Implements 44 features, including Shots on Target (SoT), corner pressure, conversion rates, and defensive solidity.
- **Leakage Prevention**: Replaced static encoders with a **Recursive Cumulative Pedigree** tracker and chronological splits, ensuring the model never "peeks" into future seasonal performance.
- **Hybrid Ensemble Training**: 
    - **CatBoost**: Optimized via **Bayesian Search** for superior handling of categorical strength gaps.
    - **PyTorch Neural Network**: A custom `TabularNN` architecture with **Cyclical Date Encoding** (Sin/Cos) to capture seasonal football patterns.
    - **Weighted Soft Voting**: A tuned ensemble that blends probabilities to maximize both raw Accuracy and Draw Recall.
- **Deployment (CI/CD)**: Uses GitHub Actions to automatically trigger deployments to an **AWS EC2** instance, ensuring the live app is always in sync with the latest code.

# ✨ Key Features

- **Ensemble Predictions**: Real-time Win/Draw/Away Win probabilities calculated by a CatBoost & Neural Net hybrid stack.
- **Match Balance Analysis**: Visualizes the "Strength Gap" and "Overall Class" of opponents to explain the model's logic.
- **Performance Radars**: Interactive Plotly charts comparing Attack Strength, Defense Solidity, and Venue Form.
- **Dynamic Momentum**: Analyzes the last 5 games of "Underlying Threat" (SoT/SoT Conceded) rather than just points.

# 🛠️ Tech Stack

- **Core**: Python 3.13, **PyTorch**, **CatBoost**, Scikit-learn, Pandas, NumPy
- **Optimization**: **Scikit-optimize** (Bayesian Search)
- **Ensembling**: **Skorch** (Scikit-learn wrapper for PyTorch)
- **Frontend**: Streamlit & Plotly
- **DevOps**: GitHub Actions, AWS EC2, Docker, Systemd

# 🚀 Quick Start

### 1. Clone & Setup

git clone [https://github.com/TanishS2003/end-to-end_ml_project.git](https://github.com/TanishS2003/end-to-end_ml_project.git)
cd end-to-end_ml_project
pip install -r requirements.txt


2. Train & Run

# Retrain the model and generate artifacts
python train_pipeline.py

# Launch the web app
streamlit run app.py
