"""
Prediction Pipeline - Final
"""

import sys
import os
import pandas as pd
import numpy as np
import torch
from datetime import datetime
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object

# IMPORTANT: Ensure blueprints are imported so pickle can load the PyTorch/Voting model
from src.utils import TabularNN, WeightedVotingWrapper


class PredictPipeline:
    def __init__(self):
        self.model_path = os.path.join('artifacts', 'model.pkl')
        self.preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')

    def predict(self, features):
        try:
            logging.info('Making prediction')

            model = load_object(self.model_path)
            preprocessor = load_object(self.preprocessor_path)

            # 1. Transform features using the exact column order
            data_scaled = preprocessor.transform(features)

            # 2. Force float32 for PyTorch compatibility
            data_ready = data_scaled.astype(np.float32)

            # 3. Predict
            prediction = model.predict(data_ready)
            probabilities = model.predict_proba(data_ready)

            result_map = {0: 'H', 1: 'D', 2: 'A'}

            # Ensure we get a scalar index
            pred_idx = int(np.ravel(prediction)[0])
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
                 home_gd: float = 0.0, home_h_pts_mp: float = 1.5,
                 home_gls: float = 0.0, home_ga: float = 0.0, home_mp: int = 1,
                 # --- NEW HOME METRICS ---
                 home_sot_mp: float = 4.0, home_sot_conceded_mp: float = 4.0,
                 home_corners_mp: float = 4.5, home_cards_mp: float = 2.0,
                 home_conversion: float = 0.3,

                 away_rank: int = 10, away_pts_mp: float = 1.5,
                 away_gd: float = 0.0, away_a_pts_mp: float = 1.5,
                 away_gls: float = 0.0, away_ga: float = 0.0, away_mp: int = 1,
                 # --- NEW AWAY METRICS ---
                 away_sot_mp: float = 4.0, away_sot_conceded_mp: float = 4.0,
                 away_corners_mp: float = 4.5, away_cards_mp: float = 2.0,
                 away_conversion: float = 0.3,

                 home_form_pts: int = 7, home_form_gf: int = 2, home_form_ga: int = 1,
                 home_form_sot: int = 20, home_form_sot_c: int = 20,  # NEW FORM
                 away_form_pts: int = 7, away_form_gf: int = 2, away_form_ga: int = 1,
                 away_form_sot: int = 20, away_form_sot_c: int = 20):  # NEW FORM

        self.home_team = home_team
        self.away_team = away_team
        self.home_rank = home_rank
        self.home_pts_mp = home_pts_mp
        self.home_gd = home_gd
        self.home_h_pts_mp = home_h_pts_mp
        self.home_gls = home_gls
        self.home_ga = home_ga
        self.home_mp = home_mp
        self.home_sot_mp = home_sot_mp
        self.home_sot_conceded_mp = home_sot_conceded_mp
        self.home_corners_mp = home_corners_mp
        self.home_cards_mp = home_cards_mp
        self.home_conversion = home_conversion

        self.away_rank = away_rank
        self.away_pts_mp = away_pts_mp
        self.away_gd = away_gd
        self.away_a_pts_mp = away_a_pts_mp
        self.away_gls = away_gls
        self.away_ga = away_ga
        self.away_mp = away_mp
        self.away_sot_mp = away_sot_mp
        self.away_sot_conceded_mp = away_sot_conceded_mp
        self.away_corners_mp = away_corners_mp
        self.away_cards_mp = away_cards_mp
        self.away_conversion = away_conversion

        self.home_form_pts = home_form_pts
        self.home_form_gf = home_form_gf
        self.home_form_ga = home_form_ga
        self.home_form_sot = home_form_sot
        self.home_form_sot_c = home_form_sot_c

        self.away_form_pts = away_form_pts
        self.away_form_gf = away_form_gf
        self.away_form_ga = away_form_ga
        self.away_form_sot = away_form_sot
        self.away_form_sot_c = away_form_sot_c

    def get_data_as_dataframe(self):
        try:
            from datetime import datetime
            current_date = datetime.now().strftime('%Y-%m-%d')

            # Strength uses the provided normalized per-match values directly
            home_strength = (self.home_pts_mp * 0.5) + \
                (self.home_gd * 0.3) + (self.home_gls * 0.2)
            away_strength = (self.away_pts_mp * 0.5) + \
                (self.away_gd * 0.3) + (self.away_gls * 0.2)

            strength_diff = home_strength - away_strength
            form_diff = self.home_form_pts - self.away_form_pts
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

                # --- NEW UNDERLYING METRICS ---
                'Home_SoT_MP': [self.home_sot_mp],
                'Away_SoT_MP': [self.away_sot_mp],
                'Home_SoT_Conceded_MP': [self.home_sot_conceded_mp],
                'Away_SoT_Conceded_MP': [self.away_sot_conceded_mp],
                'Home_Corners_MP': [self.home_corners_mp],
                'Away_Corners_MP': [self.away_corners_mp],
                'Home_Cards_MP': [self.home_cards_mp],
                'Away_Cards_MP': [self.away_cards_mp],
                'Home_Conversion': [self.home_conversion],
                'Away_Conversion': [self.away_conversion],

                'Home_Strength': [home_strength],
                'Away_Strength': [away_strength],
                'Strength_Diff': [strength_diff],
                'Rank_Gap': [rank_gap],
                'Strength_Ratio': [strength_ratio],
                'Is_Close_Match': [is_close_match],

                'Home_Form_Pts': [self.home_form_pts],
                'Home_Form_GF': [self.home_form_gf],
                'Home_Form_GA': [self.home_form_ga],
                'Home_Form_SoT': [self.home_form_sot],
                'Home_Form_SoT_C': [self.home_form_sot_c],

                'Away_Form_Pts': [self.away_form_pts],
                'Away_Form_GF': [self.away_form_gf],
                'Away_Form_GA': [self.away_form_ga],
                'Away_Form_SoT': [self.away_form_sot],
                'Away_Form_SoT_C': [self.away_form_sot_c],

                'Form_Diff': [form_diff]
            }
            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)
