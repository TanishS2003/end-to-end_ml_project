"""
Model Trainer - Final Production Version
Imports architecture from utils and uses Bayesian optimization for all stages.
"""

import warnings
import sys
import os
import time
import numpy as np
import pandas as pd
import logging as py_logging

# Sklearn & Optimization
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_sample_weight, compute_class_weight
from sklearn.metrics import classification_report
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical

# Models & PyTorch Integration
from catboost import CatBoostClassifier
import torch
import torch.nn as nn
import torch.optim as optim
from skorch import NeuralNetClassifier

# Project Imports - Blueprints now imported from utils
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, TabularNN, WeightedVotingWrapper


# Hardware acceleration
from sklearnex import patch_sklearn
patch_sklearn()

py_logging.getLogger("sklearnex").setLevel(py_logging.WARNING)
warnings.filterwarnings('ignore')


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")
    model_comparison_path: str = os.path.join(
        "artifacts", "model_comparison.csv")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        self.results = {}

    def train_single_model(self, name, model, params, n_iter, X_train, y_train, X_test, y_test):
        try:
            logging.info(f'\n{"="*70}\nTRAINING: {name}\n{"="*70}')
            start = time.time()

            # Bayesian Search optimization
            bayes_search = BayesSearchCV(
                model, params, n_iter=n_iter, cv=5,
                scoring='accuracy', n_jobs=-1, random_state=42, verbose=0
            )

            # Handle sample weights for CatBoost
            if name == 'CatBoost':
                weights = compute_sample_weight(
                    class_weight='balanced', y=y_train)
                bayes_search.fit(X_train, y_train, sample_weight=weights)
            else:
                bayes_search.fit(X_train, y_train)

            best_estimator = bayes_search.best_estimator_
            train_time = time.time() - start
            test_acc = best_estimator.score(X_test, y_test)

            logging.info(
                f'✓ Completed in {train_time:.2f}s | Test Accuracy: {test_acc:.4f}')

            y_pred = best_estimator.predict(X_test)
            report = classification_report(y_test, y_pred, target_names=[
                                           'Home', 'Draw', 'Away'], zero_division=0)
            logging.info(f'\n{report}')

            from sklearn.metrics import precision_recall_fscore_support
            _, recall, _, _ = precision_recall_fscore_support(
                y_test, y_pred, average=None, zero_division=0)

            return {
                'name': name, 'model': best_estimator, 'test_acc': test_acc,
                'cv_score': bayes_search.best_score_,
                'home_recall': recall[0], 'draw_recall': recall[1], 'away_recall': recall[2],
                'time': train_time
            }
        except Exception as e:
            logging.error(f'Error training {name}: {str(e)}')
            raise CustomException(e, sys)

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info('\n' + '='*70 + '\nMODEL TRAINING STARTED\n' + '='*70)

            # PyTorch strictly requires float32
            X_train = train_array[:, :-1].astype(np.float32)
            y_train = train_array[:, -1].astype(np.int64)
            X_test = test_array[:, :-1].astype(np.float32)
            y_test = test_array[:, -1].astype(np.int64)

            # Compute PyTorch class weights for Draw/Away priority
            class_weights = compute_class_weight(
                class_weight='balanced',
                classes=np.unique(y_train),
                y=y_train
            )
            class_weights_tensor = torch.tensor(
                class_weights, dtype=torch.float)

            # --- Stage 1: Base Models ---
            models_config = {
                'CatBoost': (
                    CatBoostClassifier(
                        random_state=42, verbose=False, allow_writing_files=False),
                    {'iterations': Integer(100, 500), 'depth': Integer(
                        3, 8), 'learning_rate': Real(0.01, 0.1, prior='log-uniform')},
                    20
                ),
                'PyTorch_NN': (
                    NeuralNetClassifier(
                        module=TabularNN, module__input_dim=X_train.shape[1],
                        criterion=nn.CrossEntropyLoss, criterion__weight=class_weights_tensor,
                        optimizer=optim.Adam, max_epochs=50, verbose=0, train_split=None, iterator_train__drop_last=True
                    ),
                    {'module__neurons': Categorical([64, 128]), 'lr': Real(
                        1e-4, 1e-2, prior='log-uniform'), 'batch_size': Categorical([16, 32])},
                    15
                )
            }

            for name, (model, params, n_iter) in models_config.items():
                self.results[name] = self.train_single_model(
                    name, model, params, n_iter, X_train, y_train, X_test, y_test)

            # --- Stage 2: Tuned Ensembles ---
            best_cat = self.results['CatBoost']['model']
            best_nn = self.results['PyTorch_NN']['model']

            # Tuned Stacking
            stack_model = StackingClassifier(
                estimators=[('cat', best_cat), ('nn', best_nn)],
                final_estimator=LogisticRegression(
                    class_weight='balanced', max_iter=1000)
            )
            self.results['Context_Stacking'] = self.train_single_model(
                'Context_Stacking', stack_model, {'final_estimator__C': Real(
                    0.1, 10.0)}, 10, X_train, y_train, X_test, y_test
            )

            # Tuned Soft Voting
            voting_wrapper = WeightedVotingWrapper(
                estimators=[('cat', best_cat), ('nn', best_nn)])
            self.results['Soft_Voting'] = self.train_single_model(
                'Soft_Voting', voting_wrapper, {'nn_weight': Real(
                    0.5, 3.0)}, 10, X_train, y_train, X_test, y_test
            )

            # --- Final Champion Selection ---
            # Prioritize Draw Recall (50%) alongside Test Accuracy (50%)
            best = max(self.results.values(), key=lambda x: (
                x['test_acc'] * 0.5) + (x['draw_recall'] * 0.5))

            logging.info(
                f'\n🥇 CHAMPION SELECTED: {best["name"]} (Acc: {best["test_acc"]:.2%}, Draw Recall: {best["draw_recall"]:.2%})')

            # Save comparison for reference
            comparison_df = pd.DataFrame(self.results).T.drop('model', axis=1)
            comparison_df.to_csv(
                self.model_trainer_config.model_comparison_path)

            # Save the winning model
            save_object(
                self.model_trainer_config.trained_model_file_path, best['model'])

            return best['test_acc']

        except Exception as e:
            raise CustomException(e, sys)
