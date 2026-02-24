"""
Model Trainer - Trains 4 models and selects best with balanced sample weights
"""

import warnings
import sys
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import accuracy_score, classification_report
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
import time
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
import logging as py_logging
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
            logging.info(f'\n{"="*70}')
            logging.info(f'TRAINING: {name}')
            logging.info(f'{"="*70}')

            # Calculate sample weights to force model to learn draws
            weights = compute_sample_weight(class_weight='balanced', y=y_train)

            start = time.time()

            bayes_search = BayesSearchCV(
                model, params, n_iter=n_iter, cv=5,
                scoring='accuracy', n_jobs=-1, random_state=42, verbose=0
            )

            # Apply weights during fitting to fix the bias
            if name == 'CatBoost':
                bayes_search.fit(X_train, y_train,
                                 sample_weight=weights, verbose=False)
            else:
                bayes_search.fit(X_train, y_train, sample_weight=weights)

            train_time = time.time() - start
            test_acc = bayes_search.score(X_test, y_test)

            logging.info(f'✓ Completed in {train_time:.2f}s')
            logging.info(f'Test Accuracy: {test_acc:.4f}')

            y_pred = bayes_search.predict(X_test)
            report = classification_report(y_test, y_pred, target_names=[
                                           'Home', 'Draw', 'Away'], zero_division=0)
            logging.info(f'\n{report}')

            from sklearn.metrics import precision_recall_fscore_support
            _, recall, _, _ = precision_recall_fscore_support(
                y_test, y_pred, average=None, zero_division=0)

            return {
                'name': name,
                'model': bayes_search.best_estimator_,
                'test_acc': test_acc,
                'cv_score': bayes_search.best_score_,
                'away_recall': recall[2],
                'time': train_time
            }

        except Exception as e:
            logging.error(f'Error training {name}: {str(e)}')
            raise CustomException(e, sys)

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info('\n' + '='*70)
            logging.info('MODEL TRAINING STARTED')
            logging.info('='*70)

            X_train = train_array[:, :-1]
            y_train = train_array[:, -1]
            X_test = test_array[:, :-1]
            y_test = test_array[:, -1]

            logging.info(
                f'Training data: {X_train.shape[0]} samples, {X_train.shape[1]} features')
            logging.info(f'Test data: {X_test.shape[0]} samples')

            models = {
                'RandomForest': (
                    RandomForestClassifier(random_state=42, n_jobs=-1),
                    {
                        'n_estimators': Integer(100, 500),
                        'max_depth': Integer(3, 10),
                        'min_samples_leaf': Integer(10, 40),
                        'max_features': Categorical(['sqrt', 'log2'])
                    },
                    32
                ),
                'XGBoost': (
                    XGBClassifier(random_state=42, n_jobs=-1,
                                  eval_metric='mlogloss', use_label_encoder=False),
                    {
                        'n_estimators': Integer(100, 500),
                        'max_depth': Integer(3, 8),
                        'learning_rate': Real(0.01, 0.1, prior='log-uniform'),
                        'subsample': Real(0.5, 0.9),
                        'colsample_bytree': Real(0.3, 0.8),
                        'reg_alpha': Real(1e-3, 10.0, prior='log-uniform'),
                        'reg_lambda': Real(1e-3, 10.0, prior='log-uniform')
                    },
                    40
                ),
                'LightGBM': (
                    LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1),
                    {
                        'n_estimators': Integer(100, 500),
                        'max_depth': Integer(3, 8),
                        'learning_rate': Real(0.01, 0.1, prior='log-uniform'),
                        'num_leaves': Integer(10, 40),
                        'colsample_bytree': Real(0.3, 0.8),
                        'reg_alpha': Real(1e-3, 10.0, prior='log-uniform'),
                        'reg_lambda': Real(1e-3, 10.0, prior='log-uniform')
                    },
                    40
                ),
                'CatBoost': (
                    CatBoostClassifier(
                        random_state=42,
                        verbose=False,
                        thread_count=-1,
                        allow_writing_files=False
                    ),
                    {
                        'iterations': Integer(100, 500),
                        'depth': Integer(3, 8),
                        'learning_rate': Real(0.01, 0.1, prior='log-uniform'),
                        'l2_leaf_reg': Real(1.0, 20.0, prior='log-uniform'),
                        'rsm': Real(0.3, 0.8)
                    },
                    40
                )
            }

            for name, (model, params, n_iter) in models.items():
                result = self.train_single_model(
                    name, model, params, n_iter, X_train, y_train, X_test, y_test)
                self.results[name] = result

            comparison = pd.DataFrame([{
                'Model': r['name'],
                'Test_Acc': f"{r['test_acc']:.4f}",
                'CV_Score': f"{r['cv_score']:.4f}",
                'Away_Recall': f"{r['away_recall']:.4f}",
                'Time': f"{r['time']:.1f}s"
            } for r in self.results.values()])

            comparison.to_csv(
                self.model_trainer_config.model_comparison_path, index=False)

            best = max(self.results.values(), key=lambda x: x['test_acc'])
            best_model = best['model']

            try:
                raw_train_df = pd.read_csv(
                    'artifacts/train.csv').drop('Result', axis=1)
                feature_names = []
                for col in raw_train_df.columns:
                    if col == 'Date':
                        feature_names.extend(['Year', 'Month', 'DayOfWeek'])
                    else:
                        feature_names.append(col)

                if hasattr(best_model, 'feature_importances_'):
                    importances = best_model.feature_importances_
                    if len(feature_names) == len(importances):
                        feature_imp_df = pd.DataFrame({
                            'Feature': feature_names,
                            'Importance': importances
                        }).sort_values(by='Importance', ascending=False)
                        feature_imp_df.to_csv(
                            'artifacts/feature_importance.csv', index=False)

                        print("\n" + "="*70)
                        print("🔝 TOP 10 PREDICTIVE FEATURES")
                        print("="*70)
                        print(feature_imp_df.head(10).to_string(index=False))
                        print("="*70 + "\n")
            except Exception as e:
                logging.warning(
                    f"Could not extract feature importance: {str(e)}")

            save_object(
                self.model_trainer_config.trained_model_file_path, best_model)

            try:
                test_df = pd.read_csv('artifacts/test.csv')
                from src.utils import load_object
                from sklearn.metrics import classification_report, confusion_matrix
                import numpy as np

                model = load_object('artifacts/model.pkl')
                preprocessor = load_object('artifacts/preprocessor.pkl')

                target_mapping = {'H': 0, 'D': 1, 'A': 2}
                X_test_df = test_df.drop('Result', axis=1)
                y_test_df = test_df['Result'].map(target_mapping).values

                X_test_transformed = preprocessor.transform(X_test_df)
                predictions = model.predict(X_test_transformed)
                probabilities = model.predict_proba(X_test_transformed)

                report = classification_report(y_test_df, predictions,
                                               target_names=[
                                                   'Home Win', 'Draw', 'Away Win'],
                                               zero_division=0)

                cm = confusion_matrix(y_test_df, predictions)

                print("Classification Report:")
                print(report)
                print("\nConfusion Matrix:")
                print("              Predicted")
                print("           Home  Draw  Away")
                print(
                    f"Home        {cm[0][0]:3d}   {cm[0][1]:3d}   {cm[0][2]:3d}")
                print(
                    f"Draw        {cm[1][0]:3d}   {cm[1][1]:3d}   {cm[1][2]:3d}")
                print(
                    f"Away        {cm[2][0]:3d}   {cm[2][1]:3d}   {cm[2][2]:3d}")

                print("\n" + "="*70)
                print("📝 SAMPLE PREDICTIONS")
                print("="*70 + "\n")

                result_map = {0: 'Home Win', 1: 'Draw', 2: 'Away Win'}

                for i in range(min(5, len(test_df))):
                    actual = result_map[int(y_test_df[i])]
                    pred_idx = int(predictions[i].ravel()[0])
                    pred = result_map[pred_idx]
                    home_prob = probabilities[i][0]
                    draw_prob = probabilities[i][1]
                    away_prob = probabilities[i][2]
                    correct = "✓" if actual == pred else "✗"

                    print(
                        f"{i+1}. {test_df.iloc[i]['HomeTeam']} vs {test_df.iloc[i]['AwayTeam']}")
                    print(
                        f"   Actual: {actual:9s} | Predicted: {pred:9s} {correct}")
                    print(
                        f"   Probs: Home {home_prob:.1%} | Draw {draw_prob:.1%} | Away {away_prob:.1%}\n")

            except Exception as e:
                logging.warning(
                    f"Could not complete test evaluation: {str(e)}")

            return best['test_acc']

        except Exception as e:
            raise CustomException(e, sys)
