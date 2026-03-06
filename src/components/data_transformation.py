"""
Data Transformation Component - Bulletproof Edition
Includes: Underlying metrics, cumulative class (leak-proof), cyclical dates, and momentum.
"""

from src.utils import save_object
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
from sklearn.utils.validation import check_is_fitted
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import sys
import os
import warnings
warnings.filterwarnings('ignore')


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join(
        'artifacts', "preprocessor.pkl")


def engineer_football_features(df):
    """
    Engineer all features chronologically to strictly prevent look-ahead bias.
    Includes advanced underlying metrics (SoT, Corners, Cards).
    """
    try:
        logging.info('Starting feature engineering with underlying metrics')

        df = df.sort_values('Date').reset_index(drop=True)
        df['Date'] = pd.to_datetime(df['Date'])

        # Initialize team stats with underlying metrics
        teams = pd.concat([df['HomeTeam'], df['AwayTeam']]).unique()
        team_stats = {team: {
            'MP': 0, 'Pts': 0, 'Gls': 0, 'GA': 0, 'GD': 0,
            'H_MP': 0, 'H_Pts': 0, 'A_MP': 0, 'A_Pts': 0,
            'SoT': 0, 'SoT_Conceded': 0, 'Corners': 0, 'Cards': 0,
            'last_5_results': []
        } for team in teams}

        features_list = []

        for idx, row in df.iterrows():
            home_t = row['HomeTeam']
            away_t = row['AwayTeam']

            home_stats = team_stats[home_t]
            away_stats = team_stats[away_t]

            # Safely extract underlying stats for the current match
            h_sot = row['HST'] if pd.notna(row['HST']) else 0
            a_sot = row['AST'] if pd.notna(row['AST']) else 0
            h_cor = row['HC'] if pd.notna(row['HC']) else 0
            a_cor = row['AC'] if pd.notna(row['AC']) else 0

            h_cards = (row['HY'] if pd.notna(row['HY']) else 0) + \
                (row['HR'] if pd.notna(row['HR']) else 0)
            a_cards = (row['AY'] if pd.notna(row['AY']) else 0) + \
                (row['AR'] if pd.notna(row['AR']) else 0)

            # Skip prediction for first 3 games, but still record their stats
            if home_stats['MP'] < 3 or away_stats['MP'] < 3:
                pass
            else:
                # Calculate ranks based purely on historical points up to this date
                all_teams_sorted = sorted(team_stats.items(),
                                          key=lambda x: (
                                              x[1]['Pts'], x[1]['GD'], x[1]['Gls']),
                                          reverse=True)
                rankings = {team: i + 1 for i,
                            (team, _) in enumerate(all_teams_sorted)}

                home_mp = max(home_stats['MP'], 1)
                away_mp = max(away_stats['MP'], 1)

                home_pts_mp = home_stats['Pts'] / home_mp
                away_pts_mp = away_stats['Pts'] / away_mp

                # Cumulative Strength (replaces the leaky TeamStrengthEncoder)
                home_strength = (home_pts_mp * 0.5) + ((home_stats['GD'] / home_mp) * 0.3) + (
                    (home_stats['Gls'] / home_mp) * 0.2)
                away_strength = (away_pts_mp * 0.5) + ((away_stats['GD'] / away_mp) * 0.3) + (
                    (away_stats['Gls'] / away_mp) * 0.2)

                h_last5 = home_stats['last_5_results']
                a_last5 = away_stats['last_5_results']

                features = {
                    'Date': row['Date'],
                    'HomeTeam': home_t,
                    'AwayTeam': away_t,

                    # Per-Match Averages (Standard)
                    'Home_Rk': rankings.get(home_t, 10),
                    'Home_Pts_MP': home_pts_mp,
                    'Home_GD': home_stats['GD'] / home_mp,
                    'Home_H_Pts_MP': home_stats['H_Pts'] / max(home_stats['H_MP'], 1),
                    'Home_Gls': home_stats['Gls'] / home_mp,
                    'Home_GA': home_stats['GA'] / home_mp,

                    'Away_Rk': rankings.get(away_t, 10),
                    'Away_Pts_MP': away_pts_mp,
                    'Away_GD': away_stats['GD'] / away_mp,
                    'Away_A_Pts_MP': away_stats['A_Pts'] / max(away_stats['A_MP'], 1),
                    'Away_Gls': away_stats['Gls'] / away_mp,
                    'Away_GA': away_stats['GA'] / away_mp,

                    # Per-Match Averages (Underlying Metrics)
                    'Home_SoT_MP': home_stats['SoT'] / home_mp,
                    'Away_SoT_MP': away_stats['SoT'] / away_mp,
                    'Home_SoT_Conceded_MP': home_stats['SoT_Conceded'] / home_mp,
                    'Away_SoT_Conceded_MP': away_stats['SoT_Conceded'] / away_mp,
                    'Home_Corners_MP': home_stats['Corners'] / home_mp,
                    'Away_Corners_MP': away_stats['Corners'] / away_mp,
                    'Home_Cards_MP': home_stats['Cards'] / home_mp,
                    'Away_Cards_MP': away_stats['Cards'] / away_mp,
                    'Home_Conversion': home_stats['Gls'] / max(home_stats['SoT'], 1),
                    'Away_Conversion': away_stats['Gls'] / max(away_stats['SoT'], 1),

                    # Cumulative Strength Metrics
                    'Home_Strength': home_strength,
                    'Away_Strength': away_strength,
                    'Strength_Diff': home_strength - away_strength,
                    'Rank_Gap': rankings.get(away_t, 10) - rankings.get(home_t, 10),
                    'Strength_Ratio': home_strength / max(away_strength, 0.1),
                    'Is_Close_Match': 1 if abs(home_strength - away_strength) < 0.25 else 0,

                    # Momentum (Last 5 Games)
                    'Home_Form_Pts': sum([g['pts'] for g in h_last5]) if h_last5 else 0,
                    'Home_Form_GF': sum([g['gf'] for g in h_last5]) if h_last5 else 0,
                    'Home_Form_GA': sum([g['ga'] for g in h_last5]) if h_last5 else 0,
                    'Home_Form_SoT': sum([g['sot'] for g in h_last5]) if h_last5 else 0,
                    'Home_Form_SoT_C': sum([g['sot_c'] for g in h_last5]) if h_last5 else 0,

                    'Away_Form_Pts': sum([g['pts'] for g in a_last5]) if a_last5 else 0,
                    'Away_Form_GF': sum([g['gf'] for g in a_last5]) if a_last5 else 0,
                    'Away_Form_GA': sum([g['ga'] for g in a_last5]) if a_last5 else 0,
                    'Away_Form_SoT': sum([g['sot'] for g in a_last5]) if a_last5 else 0,
                    'Away_Form_SoT_C': sum([g['sot_c'] for g in a_last5]) if a_last5 else 0,

                    'Form_Diff': (sum([g['pts'] for g in h_last5]) if h_last5 else 0) - (sum([g['pts'] for g in a_last5]) if a_last5 else 0),
                    'Result': row['FTR']
                }
                features_list.append(features)

            # Update historical stats AFTER engineering the feature to prevent data leakage
            h_pts = 3 if row['FTR'] == 'H' else (1 if row['FTR'] == 'D' else 0)
            a_pts = 3 if row['FTR'] == 'A' else (1 if row['FTR'] == 'D' else 0)

            # Update Home
            home_stats['MP'] += 1
            home_stats['Pts'] += h_pts
            home_stats['Gls'] += row['FTHG']
            home_stats['GA'] += row['FTAG']
            home_stats['GD'] = home_stats['Gls'] - home_stats['GA']
            home_stats['H_MP'] += 1
            home_stats['H_Pts'] += h_pts
            home_stats['SoT'] += h_sot
            home_stats['SoT_Conceded'] += a_sot
            home_stats['Corners'] += h_cor
            home_stats['Cards'] += h_cards

            home_stats['last_5_results'].append({
                'pts': h_pts, 'gf': row['FTHG'], 'ga': row['FTAG'],
                'sot': h_sot, 'sot_c': a_sot
            })
            if len(home_stats['last_5_results']) > 5:
                home_stats['last_5_results'].pop(0)

            # Update Away
            away_stats['MP'] += 1
            away_stats['Pts'] += a_pts
            away_stats['Gls'] += row['FTAG']
            away_stats['GA'] += row['FTHG']
            away_stats['GD'] = away_stats['Gls'] - away_stats['GA']
            away_stats['A_MP'] += 1
            away_stats['A_Pts'] += a_pts
            away_stats['SoT'] += a_sot
            away_stats['SoT_Conceded'] += h_sot
            away_stats['Corners'] += a_cor
            away_stats['Cards'] += a_cards

            away_stats['last_5_results'].append({
                'pts': a_pts, 'gf': row['FTAG'], 'ga': row['FTHG'],
                'sot': a_sot, 'sot_c': h_sot
            })
            if len(away_stats['last_5_results']) > 5:
                away_stats['last_5_results'].pop(0)

        result_df = pd.DataFrame(features_list)

        logging.info('✓ Feature engineering completed')
        logging.info(f'  Features created: {len(result_df.columns)}')
        logging.info(f'  Final samples: {len(result_df)}')

        return result_df

    except Exception as e:
        logging.error(f'Error in feature engineering: {str(e)}')
        raise CustomException(e, sys)


class PreprocessingTransformer(BaseEstimator, TransformerMixin):
    """
    Handles date extraction (with cyclical encoding for NN compatibility),
    mathematical clipping, and drops raw categorical text columns.
    """

    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self

    def transform(self, X):
        check_is_fitted(self)
        X = X.copy()

        # 1. Date Feature Extraction & Cyclical Encoding
        if 'Date' in X.columns:
            X['Date'] = pd.to_datetime(X['Date'])
            X['Year'] = X['Date'].dt.year

            # Cyclical encoding for Month (1-12)
            month = X['Date'].dt.month
            X['Month_sin'] = np.sin(2 * np.pi * month / 12)
            X['Month_cos'] = np.cos(2 * np.pi * month / 12)

            # Cyclical encoding for Day of Week (0-6)
            day = X['Date'].dt.dayofweek
            X['Day_sin'] = np.sin(2 * np.pi * day / 7)
            X['Day_cos'] = np.cos(2 * np.pi * day / 7)

        # 2. Cap Conversion Rates mathematically (prevent > 100% due to own goals)
        if 'Home_Conversion' in X.columns:
            X['Home_Conversion'] = X['Home_Conversion'].clip(upper=1.0)
            X['Away_Conversion'] = X['Away_Conversion'].clip(upper=1.0)

        # 3. Drop non-numeric and obsolete columns to prevent Data Leakage & PyTorch errors
        cols_to_drop = ['Date', 'HomeTeam', 'AwayTeam']
        X = X.drop(
            columns=[c for c in cols_to_drop if c in X.columns], errors='ignore')

        return X


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            preprocessor = Pipeline(steps=[
                ('feature_processor', PreprocessingTransformer()),
                ('scaler', StandardScaler())
            ])
            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, raw_data_path):
        try:
            logging.info("="*70)
            logging.info("DATA TRANSFORMATION STARTED")
            logging.info("="*70)

            df = pd.read_csv(raw_data_path)
            logging.info(f'✓ Loaded: {len(df)} rows')

            # Engineer features on FULL dataset chronologically
            df = engineer_football_features(df)

            # Chronological Split
            train_size = int(0.8 * len(df))
            train_df = df.iloc[:train_size].copy()
            test_df = df.iloc[train_size:].copy()

            logging.info(
                f'✓ Split: Train={len(train_df)}, Test={len(test_df)}')

            # Save splits
            train_df.to_csv('artifacts/train.csv', index=False)
            test_df.to_csv('artifacts/test.csv', index=False)

            # Separate features and target
            target_column = 'Result'
            target_mapping = {'H': 0, 'D': 1, 'A': 2}

            X_train = train_df.drop(target_column, axis=1)
            y_train = train_df[target_column].map(target_mapping)

            X_test = test_df.drop(target_column, axis=1)
            y_test = test_df[target_column].map(target_mapping)

            logging.info('Creating preprocessing pipeline...')
            preprocessor = self.get_data_transformer_object()

            logging.info('Fitting and transforming training data...')
            X_train_transformed = preprocessor.fit_transform(X_train, y_train)

            logging.info('Transforming test data...')
            X_test_transformed = preprocessor.transform(X_test)

            logging.info('Creating final arrays...')
            train_arr = np.c_[X_train_transformed, y_train.values]
            test_arr = np.c_[X_test_transformed, y_test.values]

            logging.info('Saving preprocessor...')
            save_object(
                self.data_transformation_config.preprocessor_obj_file_path, preprocessor)

            logging.info("DATA TRANSFORMATION COMPLETED")
            logging.info("="*70)

            return train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path

        except Exception as e:
            logging.error(f'Error in data transformation: {str(e)}')
            raise CustomException(e, sys)
