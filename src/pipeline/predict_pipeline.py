"""
Prediction Pipeline
"""

import sys
import os
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        self.model_path = os.path.join('artifacts', 'model.pkl')
        self.preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')

    def predict(self, features):
        try:
            logging.info('Making prediction')

            model = load_object(self.model_path)
            preprocessor = load_object(self.preprocessor_path)

            data_scaled = preprocessor.transform(features)
            prediction = model.predict(data_scaled)
            probabilities = model.predict_proba(data_scaled)

            result_map = {0: 'H', 1: 'D', 2: 'A'}
            # Ensure we get a scalar index
            pred_idx = int(prediction.ravel()[0])
            predicted_result = result_map[pred_idx]

            prob_dict = {
                'Home Win': float(probabilities[0][0]),
                'Draw': float(probabilities[0][1]),
                'Away Win': float(probabilities[0][2])
            }

            logging.info(f'Prediction: {predicted_result}')

            return predicted_result, prob_dict

        except Exception as e:
            logging.error(f'Error: {str(e)}')
            raise CustomException(e, sys)


class CustomData:
    def __init__(self, home_team: str, away_team: str,
                 home_rank: int = 10, home_pts_mp: float = 1.5,
                 home_gd: int = 0, home_h_pts_mp: float = 1.5,
                 home_gls: int = 0, home_ga: int = 0,
                 away_rank: int = 10, away_pts_mp: float = 1.5,
                 away_gd: int = 0, away_a_pts_mp: float = 1.5,
                 away_gls: int = 0, away_ga: int = 0):

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
        try:
            from datetime import datetime
            current_date = datetime.now().strftime('%Y-%m-%d')

            # Standard strength calculations matching transformation.py
            home_strength = (self.home_pts_mp * 0.5) + \
                ((self.home_gd / 10) * 0.3) + ((self.home_gls / 10) * 0.2)
            away_strength = (self.away_pts_mp * 0.5) + \
                ((self.away_gd / 10) * 0.3) + ((self.away_gls / 10) * 0.2)
            strength_diff = home_strength - away_strength

            # Anti-Bias calculations
            rank_gap = self.away_rank - self.home_rank
            strength_ratio = home_strength / max(away_strength, 0.1)
            is_close_match = 1 if abs(strength_diff) < 0.25 else 0

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
                'Away_GA': [self.away_ga],
                'Home_Strength': [home_strength],
                'Away_Strength': [away_strength],
                'Strength_Diff': [strength_diff],
                'Home_Form_Pts': [7],
                'Home_Form_GF': [2],
                'Home_Form_GA': [1],
                'Away_Form_Pts': [7],
                'Away_Form_GF': [2],
                'Away_Form_GA': [1],
                'Form_Diff': [0],
                'Rank_Gap': [rank_gap],
                'Strength_Ratio': [strength_ratio],
                'Is_Close_Match': [is_close_match]
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)
