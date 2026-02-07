import os
import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


# Custom Transformer: Extract temporal features from Date
class DateFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Extract temporal features from Date column
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X['Date'] = pd.to_datetime(X['Date'])
        X['Year'] = X['Date'].dt.year
        X['Month'] = X['Date'].dt.month
        X['DayOfWeek'] = X['Date'].dt.dayofweek
        return X.drop('Date', axis=1)


# Custom Transformer: Encode teams based on historical performance
class TeamStrengthEncoder(BaseEstimator, TransformerMixin):
    """
    Encode teams based on their historical points performance
    """

    def __init__(self, cols):
        self.cols = cols
        self.team_stats_map = {}
        self.global_mean = 0

    def fit(self, X, y):
        # Calculate average strength per team using target
        temp_df = pd.DataFrame({
            'team': pd.concat([X[self.cols[0]], X[self.cols[1]]]),
            'points': np.tile(y, 2)
        })
        self.team_stats_map = temp_df.groupby(
            'team')['points'].mean().to_dict()
        self.global_mean = temp_df['points'].mean()
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.cols:
            X[col] = X[col].map(self.team_stats_map).fillna(self.global_mean)
        return X


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def process_football_features(self, df):
        """
        Creates football-specific features from raw match data.
        This creates running statistics for each team.
        """
        logging.info('Starting football feature engineering')

        # Convert date to datetime and sort
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
        df = df.sort_values('Date').reset_index(drop=True)

        # Get unique teams
        teams = pd.concat([df['HomeTeam'], df['AwayTeam']]).unique()
        logging.info(f'Found {len(teams)} unique teams')

        # Initialize team statistics
        team_stats = {team: {
            'MP': 0, 'Pts': 0, 'Gls': 0, 'GA': 0,
            'H_MP': 0, 'H_Pts': 0, 'A_MP': 0, 'A_Pts': 0
        } for team in teams}

        processed_rows = []

        for index, row in df.iterrows():
            home_t = row['HomeTeam']
            away_t = row['AwayTeam']

            # Calculate current rankings (before this match)
            current_table = []
            for t, s in team_stats.items():
                current_table.append({
                    'Team': t,
                    'Pts': s['Pts'],
                    'GD': s['Gls'] - s['GA'],
                    'Gls': s['Gls']
                })

            # Sort to get ranks (Points > GD > Goals)
            rank_df = pd.DataFrame(current_table).sort_values(
                by=['Pts', 'GD', 'Gls'], ascending=False
            ).reset_index(drop=True)
            rank_df['Rk'] = rank_df.index + 1

            # Get current stats
            h_stat = team_stats[home_t]
            a_stat = team_stats[away_t]

            # Create match features
            match_features = {
                'Date': row['Date'],
                'HomeTeam': home_t,
                'AwayTeam': away_t,
                'Result': row['FTR'],  # H, D, A

                # Home Team Pre-Match Features
                'Home_Rk': rank_df.loc[rank_df['Team'] == home_t, 'Rk'].values[0],
                'Home_Pts_MP': h_stat['Pts'] / h_stat['MP'] if h_stat['MP'] > 0 else 0,
                'Home_GD': h_stat['Gls'] - h_stat['GA'],
                'Home_H_Pts_MP': h_stat['H_Pts'] / h_stat['H_MP'] if h_stat['H_MP'] > 0 else 0,
                'Home_Gls': h_stat['Gls'],
                'Home_GA': h_stat['GA'],

                # Away Team Pre-Match Features
                'Away_Rk': rank_df.loc[rank_df['Team'] == away_t, 'Rk'].values[0],
                'Away_Pts_MP': a_stat['Pts'] / a_stat['MP'] if a_stat['MP'] > 0 else 0,
                'Away_GD': a_stat['Gls'] - a_stat['GA'],
                'Away_A_Pts_MP': a_stat['A_Pts'] / a_stat['A_MP'] if a_stat['A_MP'] > 0 else 0,
                'Away_Gls': a_stat['Gls'],
                'Away_GA': a_stat['GA']
            }

            # Only add if both teams played >= 3 games (reduce volatility)
            if h_stat['MP'] >= 3 and a_stat['MP'] >= 3:
                processed_rows.append(match_features)

            # Update stats AFTER recording (to prevent data leakage)
            # Update Home Team
            h_pts = 3 if row['FTR'] == 'H' else (1 if row['FTR'] == 'D' else 0)
            team_stats[home_t]['MP'] += 1
            team_stats[home_t]['Pts'] += h_pts
            team_stats[home_t]['Gls'] += row['FTHG']
            team_stats[home_t]['GA'] += row['FTAG']
            team_stats[home_t]['H_MP'] += 1
            team_stats[home_t]['H_Pts'] += h_pts

            # Update Away Team
            a_pts = 3 if row['FTR'] == 'A' else (1 if row['FTR'] == 'D' else 0)
            team_stats[away_t]['MP'] += 1
            team_stats[away_t]['Pts'] += a_pts
            team_stats[away_t]['Gls'] += row['FTAG']
            team_stats[away_t]['GA'] += row['FTHG']
            team_stats[away_t]['A_MP'] += 1
            team_stats[away_t]['A_Pts'] += a_pts

        result_df = pd.DataFrame(processed_rows)
        logging.info(
            f'Feature engineering completed: {len(result_df)} samples created')

        return result_df

    def get_data_transformer_object(self):
        """
        This function creates the preprocessing pipeline with custom transformers
        """
        try:
            logging.info('Creating preprocessing pipeline')

            # Define the pipeline with custom transformers
            preprocessing_pipeline = Pipeline(
                steps=[
                    ('date_features', DateFeatureExtractor()),
                    ('team_strength', TeamStrengthEncoder(
                        cols=['HomeTeam', 'AwayTeam']))
                ]
            )

            logging.info('Preprocessing pipeline created successfully')

            return preprocessing_pipeline

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            logging.info('Entered data transformation method')

            # Read train and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info('Read train and test data completed')
            logging.info(f'Train data shape: {train_df.shape}')
            logging.info(f'Test data shape: {test_df.shape}')

            # Apply football feature engineering
            logging.info('Applying feature engineering to train data')
            train_df = self.process_football_features(train_df)

            logging.info('Applying feature engineering to test data')
            test_df = self.process_football_features(test_df)

            # Get preprocessing object
            logging.info('Obtaining preprocessing object')
            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = 'Result'

            # Separate features and target
            # Map target: H -> 0, D -> 1, A -> 2
            target_mapping = {'H': 0, 'D': 1, 'A': 2}

            input_feature_train_df = train_df.drop(
                columns=[target_column_name])
            target_feature_train_df = train_df[target_column_name].map(
                target_mapping)

            input_feature_test_df = test_df.drop(columns=[target_column_name])
            target_feature_test_df = test_df[target_column_name].map(
                target_mapping)

            logging.info(
                'Applying preprocessing on training and test dataframes')

            # Fit on train, transform both
            input_feature_train_arr = preprocessing_obj.fit_transform(
                input_feature_train_df,
                target_feature_train_df
            )
            input_feature_test_arr = preprocessing_obj.transform(
                input_feature_test_df)

            # Combine features and target
            train_arr = np.c_[
                input_feature_train_arr,
                np.array(target_feature_train_df)
            ]
            test_arr = np.c_[
                input_feature_test_arr,
                np.array(target_feature_test_df)
            ]

            logging.info(f'Train array shape: {train_arr.shape}')
            logging.info(f'Test array shape: {test_arr.shape}')

            # Save preprocessing object
            logging.info('Saving preprocessing object')
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            logging.info('Data transformation completed')

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    from src.components.data_ingestion import DataIngestion

    # First run data ingestion
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    # Then run data transformation
    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(
        train_data, test_data)

    print(f"Train array shape: {train_arr.shape}")
    print(f"Test array shape: {test_arr.shape}")
