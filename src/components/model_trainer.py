import os
import sys
from dataclasses import dataclass
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from skopt import BayesSearchCV
from skopt.space import Integer, Categorical

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info('Split training and test input data')

            # Split features and target
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            logging.info(f'X_train shape: {X_train.shape}')
            logging.info(f'X_test shape: {X_test.shape}')

            # Define models to train
            models = {
                "Random Forest": RandomForestClassifier(random_state=42),
            }

            # Define hyperparameter search space for Bayesian Optimization
            params = {
                "Random Forest": {
                    'n_estimators': Integer(100, 500),
                    'max_depth': Integer(5, 20),
                    'min_samples_leaf': Integer(5, 30),
                    'max_features': Categorical(['sqrt', 'log2', None])
                }
            }

            logging.info('Starting model training with Bayesian Optimization')

            # Bayesian Optimization for Random Forest
            model_name = "Random Forest"
            model = models[model_name]
            param_space = params[model_name]

            logging.info(f'Training {model_name} with Bayesian optimization')

            # Initialize BayesSearchCV
            bayes_search = BayesSearchCV(
                estimator=model,
                search_spaces=param_space,
                n_iter=32,              # Number of parameter settings sampled
                cv=3,                   # 3-fold cross-validation
                scoring='accuracy',
                n_jobs=-1,              # Use all available cores
                random_state=42,
                verbose=1
            )

            # Fit the Bayesian search
            bayes_search.fit(X_train, y_train)

            # Get the best model
            best_model = bayes_search.best_estimator_

            logging.info(f'Best parameters: {bayes_search.best_params_}')
            logging.info(f'Best CV accuracy: {bayes_search.best_score_:.4f}')

            # Make predictions on train set
            logging.info('Evaluating on training set')
            y_train_pred = best_model.predict(X_train)
            train_accuracy = accuracy_score(y_train, y_train_pred)
            logging.info(f'Training Accuracy: {train_accuracy:.4f}')

            # Make predictions on test set
            logging.info('Evaluating on test set')
            y_test_pred = best_model.predict(X_test)
            test_accuracy = accuracy_score(y_test, y_test_pred)
            logging.info(f'Test Accuracy: {test_accuracy:.4f}')

            # Log classification report
            logging.info('Classification Report:')
            target_names = ['Home Win', 'Draw', 'Away Win']
            report = classification_report(
                y_test, y_test_pred, target_names=target_names)
            logging.info(f'\n{report}')

            # Log confusion matrix
            cm = confusion_matrix(y_test, y_test_pred)
            logging.info(f'Confusion Matrix:\n{cm}')

            # Check if model meets minimum accuracy threshold
            if test_accuracy < 0.40:
                raise CustomException(
                    f"No best model found with accuracy > 40%. Best accuracy: {test_accuracy:.4f}",
                    sys
                )

            logging.info(
                f'Best model found: {model_name} with test accuracy: {test_accuracy:.4f}')

            # Save the best model
            logging.info('Saving the best model')
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            logging.info('Model training completed successfully')

            return test_accuracy

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    from src.components.data_ingestion import DataIngestion
    from src.components.data_transformation import DataTransformation

    # Step 1: Data Ingestion
    logging.info('Starting Data Ingestion')
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    # Step 2: Data Transformation
    logging.info('Starting Data Transformation')
    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(
        train_data,
        test_data
    )

    # Step 3: Model Training
    logging.info('Starting Model Training')
    modeltrainer = ModelTrainer()
    accuracy = modeltrainer.initiate_model_trainer(train_arr, test_arr)

    print(f"\nFinal Test Accuracy: {accuracy:.4f}")
