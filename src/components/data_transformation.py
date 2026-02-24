"""
Data Transformation Component
Includes: Team strength, momentum (last 5 games), enhanced features
"""

from src.utils import save_object
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
from sklearn.utils.validation import check_is_fitted
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
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
    Engineer ALL features including team strength and momentum
    """
    try:
        logging.info('Starting feature engineering with ALL new features')

        df = df.sort_values('Date').reset_index(drop=True)
        df['Date'] = pd.to_datetime(df['Date'])

        # Initialize team stats
        teams = pd.concat([df['HomeTeam'], df['AwayTeam']]).unique()
        team_stats = {team: {
            'MP': 0, 'Pts': 0, 'Gls': 0, 'GA': 0, 'GD': 0,
            'H_MP': 0, 'H_Pts': 0, 'A_MP': 0, 'A_Pts': 0,
            'last_5_results': []  # NEW: Track last 5 games
        } for team in teams}

        features_list = []

        for idx, row in df.iterrows():
            home_t = row['HomeTeam']
            away_t = row['AwayTeam']

            home_stats = team_stats[home_t]
            away_stats = team_stats[away_t]

            # Skip if not enough matches
            if home_stats['MP'] < 3 or away_stats['MP'] < 3:
                # Update stats
                h_pts = 3 if row['FTR'] == 'H' else (
                    1 if row['FTR'] == 'D' else 0)
                a_pts = 3 if row['FTR'] == 'A' else (
                    1 if row['FTR'] == 'D' else 0)

                home_stats['MP'] += 1
                home_stats['Pts'] += h_pts
                home_stats['Gls'] += row['FTHG']
                home_stats['GA'] += row['FTAG']
                home_stats['GD'] = home_stats['Gls'] - home_stats['GA']
                home_stats['H_MP'] += 1
                home_stats['H_Pts'] += h_pts
                home_stats['last_5_results'].append(
                    {'pts': h_pts, 'gf': row['FTHG'], 'ga': row['FTAG']})
                if len(home_stats['last_5_results']) > 5:
                    home_stats['last_5_results'].pop(0)

                away_stats['MP'] += 1
                away_stats['Pts'] += a_pts
                away_stats['Gls'] += row['FTAG']
                away_stats['GA'] += row['FTHG']
                away_stats['GD'] = away_stats['Gls'] - away_stats['GA']
                away_stats['A_MP'] += 1
                away_stats['A_Pts'] += a_pts
                away_stats['last_5_results'].append(
                    {'pts': a_pts, 'gf': row['FTAG'], 'ga': row['FTHG']})
                if len(away_stats['last_5_results']) > 5:
                    away_stats['last_5_results'].pop(0)

                continue

            # Calculate rankings
            all_teams_sorted = sorted(team_stats.items(),
                                      key=lambda x: (
                                          x[1]['Pts'], x[1]['GD'], x[1]['Gls']),
                                      reverse=True)
            rankings = {team: idx + 1 for idx,
                        (team, _) in enumerate(all_teams_sorted)}

            # ORIGINAL FEATURES
            home_rk = rankings.get(home_t, 10)
            away_rk = rankings.get(away_t, 10)

            home_pts_mp = home_stats['Pts'] / \
                home_stats['MP'] if home_stats['MP'] > 0 else 0
            away_pts_mp = away_stats['Pts'] / \
                away_stats['MP'] if away_stats['MP'] > 0 else 0

            home_h_pts_mp = home_stats['H_Pts'] / \
                home_stats['H_MP'] if home_stats['H_MP'] > 0 else 0
            away_a_pts_mp = away_stats['A_Pts'] / \
                away_stats['A_MP'] if away_stats['A_MP'] > 0 else 0

            # NEW FEATURE 1: Overall Team Strength (venue-independent)
            home_strength = (home_pts_mp * 0.5) + ((home_stats['GD'] / max(home_stats['MP'], 1)) * 0.3) + \
                ((home_stats['Gls'] / max(home_stats['MP'], 1)) * 0.2)
            away_strength = (away_pts_mp * 0.5) + ((away_stats['GD'] / max(away_stats['MP'], 1)) * 0.3) + \
                ((away_stats['Gls'] / max(away_stats['MP'], 1)) * 0.2)

            strength_diff = home_strength - away_strength

            # NEW FEATURE 2: Last 5 Games Momentum
            home_form_pts = sum(
                [g['pts'] for g in home_stats['last_5_results']]) if home_stats['last_5_results'] else 0
            home_form_gf = sum([g['gf'] for g in home_stats['last_5_results']]
                               ) if home_stats['last_5_results'] else 0
            home_form_ga = sum([g['ga'] for g in home_stats['last_5_results']]
                               ) if home_stats['last_5_results'] else 0

            away_form_pts = sum(
                [g['pts'] for g in away_stats['last_5_results']]) if away_stats['last_5_results'] else 0
            away_form_gf = sum([g['gf'] for g in away_stats['last_5_results']]
                               ) if away_stats['last_5_results'] else 0
            away_form_ga = sum([g['ga'] for g in away_stats['last_5_results']]
                               ) if away_stats['last_5_results'] else 0

            form_diff = home_form_pts - away_form_pts

            features = {
                'Date': row['Date'],
                'HomeTeam': home_t,
                'AwayTeam': away_t,

                # Original features
                'Home_Rk': home_rk,
                'Home_Pts_MP': home_pts_mp,
                'Home_GD': home_stats['GD'],
                'Home_H_Pts_MP': home_h_pts_mp,
                'Home_Gls': home_stats['Gls'],
                'Home_GA': home_stats['GA'],

                'Away_Rk': away_rk,
                'Away_Pts_MP': away_pts_mp,
                'Away_GD': away_stats['GD'],
                'Away_A_Pts_MP': away_a_pts_mp,
                'Away_Gls': away_stats['Gls'],
                'Away_GA': away_stats['GA'],

                # NEW: Team Strength
                'Home_Strength': home_strength,
                'Away_Strength': away_strength,
                'Strength_Diff': strength_diff,

                # NEW: Momentum (Last 5 games)
                'Home_Form_Pts': home_form_pts,
                'Home_Form_GF': home_form_gf,
                'Home_Form_GA': home_form_ga,
                'Away_Form_Pts': away_form_pts,
                'Away_Form_GF': away_form_gf,
                'Away_Form_GA': away_form_ga,
                'Form_Diff': form_diff,

                # NEW ANTI-BIAS FEATURES
                'Rank_Gap': away_rk - home_rk,
                'Strength_Ratio': home_strength / max(away_strength, 0.1),
                'Is_Close_Match': 1 if abs(strength_diff) < 0.25 else 0,

                'Result': row['FTR']
            }

            features_list.append(features)

            # Update stats for next iteration
            h_pts = 3 if row['FTR'] == 'H' else (1 if row['FTR'] == 'D' else 0)
            a_pts = 3 if row['FTR'] == 'A' else (1 if row['FTR'] == 'D' else 0)

            home_stats['MP'] += 1
            home_stats['Pts'] += h_pts
            home_stats['Gls'] += row['FTHG']
            home_stats['GA'] += row['FTAG']
            home_stats['GD'] = home_stats['Gls'] - home_stats['GA']
            home_stats['H_MP'] += 1
            home_stats['H_Pts'] += h_pts
            home_stats['last_5_results'].append(
                {'pts': h_pts, 'gf': row['FTHG'], 'ga': row['FTAG']})
            if len(home_stats['last_5_results']) > 5:
                home_stats['last_5_results'].pop(0)

            away_stats['MP'] += 1
            away_stats['Pts'] += a_pts
            away_stats['Gls'] += row['FTAG']
            away_stats['GA'] += row['FTHG']
            away_stats['GD'] = away_stats['Gls'] - away_stats['GA']
            away_stats['A_MP'] += 1
            away_stats['A_Pts'] += a_pts
            away_stats['last_5_results'].append(
                {'pts': a_pts, 'gf': row['FTAG'], 'ga': row['FTHG']})
            if len(away_stats['last_5_results']) > 5:
                away_stats['last_5_results'].pop(0)

        result_df = pd.DataFrame(features_list)

        logging.info(f'✓ Feature engineering completed')
        logging.info(
            f'  Features created: {len(result_df.columns)} (includes team strength + momentum)')
        logging.info(f'  Final samples: {len(result_df)}')

        return result_df

    except Exception as e:
        logging.error(f'Error in feature engineering: {str(e)}')
        raise CustomException(e, sys)


class DateFeatureExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self

    def transform(self, X):
        check_is_fitted(self)
        X = X.copy()
        X['Date'] = pd.to_datetime(X['Date'])
        X['Year'] = X['Date'].dt.year
        X['Month'] = X['Date'].dt.month
        X['DayOfWeek'] = X['Date'].dt.dayofweek
        X = X.drop('Date', axis=1)
        return X


class TeamStrengthEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.team_encodings = {}
        self.global_mean = 0.5

    def fit(self, X, y=None):
        if y is None:
            return self

        df = X.copy().reset_index(drop=True)
        df['target'] = pd.Series(y).reset_index(drop=True)

        all_teams = []
        for _, row in df.iterrows():
            if row['target'] == 0:  # Home win
                all_teams.append({'team': row['HomeTeam'], 'points': 3})
                all_teams.append({'team': row['AwayTeam'], 'points': 0})
            elif row['target'] == 1:  # Draw
                all_teams.append({'team': row['HomeTeam'], 'points': 1})
                all_teams.append({'team': row['AwayTeam'], 'points': 1})
            else:  # Away win
                all_teams.append({'team': row['HomeTeam'], 'points': 0})
                all_teams.append({'team': row['AwayTeam'], 'points': 3})

        temp_df = pd.DataFrame(all_teams)
        if not temp_df.empty:
            team_perf = temp_df.groupby('team')['points'].mean()
            self.team_encodings = (team_perf / 3).to_dict()
            self.global_mean = temp_df['points'].mean() / 3

        self.is_fitted_ = True
        return self

    def transform(self, X):
        check_is_fitted(self)
        X = X.copy()
        X['HomeTeam'] = X['HomeTeam'].map(
            self.team_encodings).fillna(self.global_mean)
        X['AwayTeam'] = X['AwayTeam'].map(
            self.team_encodings).fillna(self.global_mean)
        return X


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            preprocessor = Pipeline(steps=[
                ('date_features', DateFeatureExtractor()),
                ('team_strength', TeamStrengthEncoder())
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

            # Engineer features on FULL dataset
            df = engineer_football_features(df)

            # Split
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

            # Preprocessing
            preprocessor = self.get_data_transformer_object()

            logging.info('Fitting and transforming training data...')
            # Using fit_transform ensures internal steps are properly fitted
            # and returns the result for training alignment
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
