import os
import sys
import pickle
import numpy as np
from sklearn.metrics import accuracy_score

from src.exception import CustomException
from src.logger import logging

import torch.nn as nn
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import VotingClassifier


def save_object(file_path, obj):
    """
    Save a Python object to a file using pickle

    Args:
        file_path: Path where the object should be saved
        obj: Python object to save
    """
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

        logging.info(f'Object saved successfully at {file_path}')

    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    """
    Load a Python object from a pickle file

    Args:
        file_path: Path to the pickle file

    Returns:
        The unpickled Python object
    """
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models, param=None):
    """
    Evaluate multiple models and return their performance scores

    Args:
        X_train: Training features
        y_train: Training target
        X_test: Test features
        y_test: Test target
        models: Dictionary of models to evaluate
        param: Dictionary of hyperparameters for each model (optional)

    Returns:
        Dictionary with model names as keys and test scores as values
    """
    try:
        report = {}

        for model_name, model in models.items():
            logging.info(f'Training {model_name}')

            # Train model
            model.fit(X_train, y_train)

            # Predict on test data
            y_test_pred = model.predict(X_test)

            # Calculate accuracy
            test_model_score = accuracy_score(y_test, y_test_pred)

            report[model_name] = test_model_score

            logging.info(
                f'{model_name} - Test Accuracy: {test_model_score:.4f}')

        return report

    except Exception as e:
        raise CustomException(e, sys)


class TabularNN(nn.Module):
    def __init__(self, input_dim=30, neurons=64, dropout_rate=0.2):
        super(TabularNN, self).__init__()
        self.layer1 = nn.Linear(input_dim, neurons)
        self.bn1 = nn.BatchNorm1d(neurons)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.layer2 = nn.Linear(neurons, neurons // 2)
        self.bn2 = nn.BatchNorm1d(neurons // 2)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.output = nn.Linear(neurons // 2, 3)

    def forward(self, X):
        X = torch.relu(self.bn1(self.layer1(X)))
        X = self.dropout1(X)
        X = torch.relu(self.bn2(self.layer2(X)))
        X = self.dropout2(X)
        X = self.output(X)
        return X


class WeightedVotingWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, estimators=None, nn_weight=1.0):
        self.estimators = estimators
        self.nn_weight = nn_weight
        self.model_ = None

    def fit(self, X, y):
        self.model_ = VotingClassifier(
            estimators=self.estimators, voting='soft', weights=[1.0, self.nn_weight])
        self.model_.fit(X, y)
        self.classes_ = self.model_.classes_
        return self

    def predict(self, X): return self.model_.predict(X)
    def predict_proba(self, X): return self.model_.predict_proba(X)
    def score(self, X, y): return self.model_.score(X, y)
