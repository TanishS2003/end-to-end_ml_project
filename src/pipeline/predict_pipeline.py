import sys
import os
import pandas as pd
import numpy as np
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object


class PredictPipeline:
    """
    Pipeline for making predictions on new match data
    """

    def __init__(self):
        logging.info('Initializing PredictPipeline')
        self.model_path = os.path.join('artifacts', 'model.pkl')
        self.preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
        logging.info(f'Model path: {self.model_path}')
        logging.info(f'Preprocessor path: {self.preprocessor_path}')

    def predict(self, features):
        """
        Make prediction on feature dataframe

        Args:
            features: DataFrame with match features

        Returns:
            prediction: 'H' (Home Win), 'D' (Draw), or 'A' (Away Win)
            probabilities: Dictionary with probabilities for each outcome
        """
        try:
            logging.info('='*70)
            logging.info('Starting prediction process')
            logging.info('='*70)
            logging.info('Loading model and preprocessor')

            # Verify files exist
            if not os.path.exists(self.model_path):
                logging.error(f'Model file not found: {self.model_path}')
                raise FileNotFoundError(
                    f'Model file not found: {self.model_path}')

            if not os.path.exists(self.preprocessor_path):
                logging.error(
                    f'Preprocessor file not found: {self.preprocessor_path}')
                raise FileNotFoundError(
                    f'Preprocessor file not found: {self.preprocessor_path}')

            # Load model and preprocessor
            model = load_object(self.model_path)
            preprocessor = load_object(self.preprocessor_path)

            logging.info('✓ Model and preprocessor loaded successfully')
            logging.info(f'Input features shape: {features.shape}')
            logging.info(f'Input columns: {list(features.columns)}')

            # Transform features
            logging.info('Transforming features with preprocessor')
            data_scaled = preprocessor.transform(features)
            logging.info(f'✓ Features transformed: {data_scaled.shape}')

            # Make prediction
            logging.info('Making prediction with model')
            prediction = model.predict(data_scaled)
            probabilities = model.predict_proba(data_scaled)

            logging.info(f'✓ Prediction made: {prediction[0]}')
            logging.info(f'Raw probabilities: {probabilities[0]}')

            # Map prediction back to result
            result_map = {0: 'H', 1: 'D', 2: 'A'}
            predicted_result = result_map[prediction[0]]

            # Create probability dictionary
            prob_dict = {
                'Home Win': float(probabilities[0][0]),
                'Draw': float(probabilities[0][1]),
                'Away Win': float(probabilities[0][2])
            }

            logging.info(f'Predicted result: {predicted_result}')
            logging.info(f'Probabilities: {prob_dict}')
            logging.info('='*70)
            logging.info('Prediction completed successfully')
            logging.info('='*70)

            return predicted_result, prob_dict

        except FileNotFoundError as e:
            logging.error(f'File not found error: {str(e)}')
            raise CustomException(e, sys)
        except Exception as e:
            logging.error(f'Error in prediction: {str(e)}')
            raise CustomException(e, sys)


class CustomData:
    """
    Class for handling custom match data input
    """

    def __init__(
        self,
        home_team: str,
        away_team: str,
        home_rank: int = 10,
        home_pts_mp: float = 1.5,
        home_gd: int = 0,
        home_h_pts_mp: float = 1.5,
        home_gls: int = 0,
        home_ga: int = 0,
        away_rank: int = 10,
        away_pts_mp: float = 1.5,
        away_gd: int = 0,
        away_a_pts_mp: float = 1.5,
        away_gls: int = 0,
        away_ga: int = 0
    ):
        """
        Initialize match data

        For a simple prediction app, you can use default stats.
        For production, these should come from your database with actual team stats.
        """
        logging.info('Creating CustomData object')
        logging.info(f'Home team: {home_team}, Away team: {away_team}')

        self.home_team = home_team
        self.away_team = away_team
        self.home_rank = home_rank
        self.home_pts_mp = home_pts_mp
        self.home_gd = home_gd
        self.home_h_pts_mp = home_h_pts_mp
        self.home_gls = home_gls
        self.home_ga = home_ga
        self.away_rank = away_rank
        self.away_pts_mp = away_pts_mp
        self.away_gd = away_gd
        self.away_a_pts_mp = away_a_pts_mp
        self.away_gls = away_gls
        self.away_ga = away_ga

    def get_data_as_dataframe(self):
        """
        Convert custom data to DataFrame format expected by model
        """
        try:
            logging.info('Converting custom data to DataFrame')

            # Create a dummy date (will be transformed to Year, Month, DayOfWeek)
            from datetime import datetime
            current_date = datetime.now().strftime('%Y-%m-%d')

            custom_data_input_dict = {
                'Date': [current_date],
                'HomeTeam': [self.home_team],
                'AwayTeam': [self.away_team],
                'Home_Rk': [self.home_rank],
                'Home_Pts_MP': [self.home_pts_mp],
                'Home_GD': [self.home_gd],
                'Home_H_Pts_MP': [self.home_h_pts_mp],
                'Home_Gls': [self.home_gls],
                'Home_GA': [self.home_ga],
                'Away_Rk': [self.away_rank],
                'Away_Pts_MP': [self.away_pts_mp],
                'Away_GD': [self.away_gd],
                'Away_A_Pts_MP': [self.away_a_pts_mp],
                'Away_Gls': [self.away_gls],
                'Away_GA': [self.away_ga]
            }

            df = pd.DataFrame(custom_data_input_dict)

            logging.info('✓ Custom data converted to DataFrame')
            logging.info(f'DataFrame shape: {df.shape}')

            return df

        except Exception as e:
            logging.error(f'Error converting data to DataFrame: {str(e)}')
            raise CustomException(e, sys)


# Simple prediction function for quick use
def predict_match(home_team: str, away_team: str):
    """
    Simple function to predict match outcome

    Args:
        home_team: Name of home team
        away_team: Name of away team

    Returns:
        prediction: 'H', 'D', or 'A'
        probabilities: Dict with probabilities
    """
    try:
        logging.info(f'predict_match called: {home_team} vs {away_team}')

        # Create custom data with default stats
        custom_data = CustomData(
            home_team=home_team,
            away_team=away_team
        )

        # Convert to DataFrame
        pred_df = custom_data.get_data_as_dataframe()

        # Make prediction
        predict_pipeline = PredictPipeline()
        prediction, probabilities = predict_pipeline.predict(pred_df)

        logging.info(f'predict_match completed: {prediction}')

        return prediction, probabilities

    except Exception as e:
        logging.error(f'Error in predict_match: {str(e)}')
        raise CustomException(e, sys)


if __name__ == "__main__":
    # Test prediction
    logging.info("="*70)
    logging.info("TESTING PREDICTION PIPELINE")
    logging.info("="*70)

    print("\n" + "="*70)
    print("TESTING PREDICTION PIPELINE")
    print("="*70 + "\n")

    # Test match: Barcelona vs Real Madrid
    home_team = "Barcelona"
    away_team = "Real Madrid"

    print(f"Match: {home_team} vs {away_team}\n")

    try:
        prediction, probabilities = predict_match(home_team, away_team)

        print(f"Predicted Result: {prediction}")
        print(f"\nProbabilities:")
        for outcome, prob in probabilities.items():
            print(f"  {outcome}: {prob:.2%}")

        print("\n" + "="*70 + "\n")

    except CustomException as e:
        print(f"CustomException occurred: {e}")
        logging.error(f"Test failed with CustomException: {e}")
    except Exception as e:
        print(f"Error: {e}")
        logging.error(f"Test failed with Exception: {e}")
