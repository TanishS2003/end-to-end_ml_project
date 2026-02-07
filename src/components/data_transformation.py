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
        logging.info('DateFeatureExtractor: Starting fit')
        logging.info(f'DateFeatureExtractor: Input shape: {X.shape}')

        # Validate input
        if 'Date' not in X.columns:
            logging.error(
                'DateFeatureExtractor: Date column not found in input')
            raise ValueError('Date column is required')

        # Set fitted attribute for sklearn compatibility
        self.n_features_in_ = X.shape[1]
        logging.info(
            f'DateFeatureExtractor: Fitted successfully with {self.n_features_in_} features')

        return self

    def transform(self, X):
        logging.info('DateFeatureExtractor: Starting transform')
        logging.info(f'DateFeatureExtractor: Input shape: {X.shape}')

        # Check if fitted
        if not hasattr(self, 'n_features_in_'):
            logging.error('DateFeatureExtractor: Not fitted yet!')
            raise ValueError(
                'DateFeatureExtractor is not fitted. Call fit() first.')

        # Validate input
        if 'Date' not in X.columns:
            logging.error(
                'DateFeatureExtractor: Date column not found in transform input')
            raise ValueError('Date column is required')

        X = X.copy()

        # Convert dates
        try:
            X['Date'] = pd.to_datetime(X['Date'])
            logging.info('DateFeatureExtractor: Date conversion successful')
        except Exception as e:
            logging.error(f'DateFeatureExtractor: Date conversion failed: {e}')
            raise

        # Extract features
        X['Year'] = X['Date'].dt.year
        X['Month'] = X['Date'].dt.month
        X['DayOfWeek'] = X['Date'].dt.dayofweek

        result = X.drop('Date', axis=1)
        logging.info(
            f'DateFeatureExtractor: Transform complete. Output shape: {result.shape}')
        logging.info(
            f'DateFeatureExtractor: New columns added: Year, Month, DayOfWeek')

        return result


# Custom Transformer: Encode teams based on historical performance
class TeamStrengthEncoder(BaseEstimator, TransformerMixin):
    """
    Encode teams based on their historical points performance
    """

    def __init__(self, cols):
        self.cols = cols
        self.team_stats_map = {}
        self.global_mean = 0

    def fit(self, X, y=None):
        logging.info('TeamStrengthEncoder: Starting fit')
        logging.info(f'TeamStrengthEncoder: Input shape: {X.shape}')
        logging.info(f'TeamStrengthEncoder: Encoding columns: {self.cols}')

        # Validate y is provided
        if y is None:
            logging.error(
                'TeamStrengthEncoder: Target variable y is required during fit')
            raise ValueError(
                "TeamStrengthEncoder requires y (target) during fit")

        logging.info(f'TeamStrengthEncoder: Target shape: {y.shape}')

        # Validate columns exist
        for col in self.cols:
            if col not in X.columns:
                logging.error(
                    f'TeamStrengthEncoder: Column {col} not found in input')
                raise ValueError(f'Column {col} not found in input DataFrame')

        # Calculate average strength per team using target
        try:
            temp_df = pd.DataFrame({
                'team': pd.concat([X[self.cols[0]], X[self.cols[1]]]),
                'points': np.tile(y, 2)
            })
            logging.info(
                f'TeamStrengthEncoder: Created temporary mapping with {len(temp_df)} rows')

            self.team_stats_map = temp_df.groupby(
                'team')['points'].mean().to_dict()
            self.global_mean = temp_df['points'].mean()

            logging.info(
                f'TeamStrengthEncoder: Encoded {len(self.team_stats_map)} unique teams')
            logging.info(
                f'TeamStrengthEncoder: Global mean: {self.global_mean:.4f}')
            logging.info(
                f'TeamStrengthEncoder: Sample encodings: {dict(list(self.team_stats_map.items())[:3])}')

        except Exception as e:
            logging.error(f'TeamStrengthEncoder: Error during encoding: {e}')
            raise

        # Set fitted attributes for sklearn compatibility
        self.n_features_in_ = X.shape[1]
        self.is_fitted_ = True

        logging.info(
            f'TeamStrengthEncoder: Fitted successfully with {self.n_features_in_} features')

        return self

    def transform(self, X):
        logging.info('TeamStrengthEncoder: Starting transform')
        logging.info(f'TeamStrengthEncoder: Input shape: {X.shape}')

        # Check if fitted
        if not hasattr(self, 'is_fitted_'):
            logging.error('TeamStrengthEncoder: Not fitted yet!')
            raise ValueError("This TeamStrengthEncoder instance is not fitted yet. "
                             "Call 'fit' with appropriate arguments before using transform.")

        logging.info('TeamStrengthEncoder: Verified transformer is fitted')
        logging.info(
            f'TeamStrengthEncoder: Using {len(self.team_stats_map)} team encodings')

        # Validate columns exist
        for col in self.cols:
            if col not in X.columns:
                logging.error(
                    f'TeamStrengthEncoder: Column {col} not found in transform input')
                raise ValueError(f'Column {col} not found in input DataFrame')

        X = X.copy()

        # Track encoding statistics
        unknown_teams = set()

        for col in self.cols:
            # Count teams before encoding
            unique_teams_before = X[col].nunique()

            # Apply encoding
            X[col] = X[col].map(self.team_stats_map).fillna(self.global_mean)

            # Track unknown teams (would use global mean)
            # Note: Can't reliably detect unknown teams after encoding

            logging.info(
                f'TeamStrengthEncoder: Encoded column {col} with {unique_teams_before} unique teams')

        logging.info(
            f'TeamStrengthEncoder: Transform complete. Output shape: {X.shape}')

        return X


@dataclass
class DataTransformationConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    preprocessor_obj_file_path: str = os.path.join(
        'artifacts', 'preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def process_football_features(self, df):
        """
        Creates football-specific features from raw match data.
        This creates running statistics for each team.

        CRITICAL: This must be called on the FULL dataset BEFORE train/test split
        to avoid data leakage!
        """
        logging.info('Starting football feature engineering')
        logging.info(f'Input data shape: {df.shape}')

        # Convert date to datetime and sort
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)
        logging.info('✓ Data sorted chronologically')

        # Get unique teams
        teams = pd.concat([df['HomeTeam'], df['AwayTeam']]).unique()
        logging.info(f'Found {len(teams)} unique teams')

        # Initialize team statistics
        team_stats = {team: {
            'MP': 0, 'Pts': 0, 'Gls': 0, 'GA': 0,
            'H_MP': 0, 'H_Pts': 0, 'A_MP': 0, 'A_Pts': 0
        } for team in teams}

        processed_rows = []
        filtered_count = 0

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
            else:
                filtered_count += 1

            # Update stats AFTER recording (to prevent data leakage within this function)
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
        logging.info(f'✓ Feature engineering completed')
        logging.info(f'  Original matches: {len(df)}')
        logging.info(f'  Filtered out (MP < 3): {filtered_count}')
        logging.info(f'  Final samples: {len(result_df)}')
        logging.info(f'  Features created: {len(result_df.columns)}')

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

            logging.info('✓ Preprocessing pipeline created successfully')
            logging.info(
                f'  Pipeline steps: {list(preprocessing_pipeline.named_steps.keys())}')

            return preprocessing_pipeline

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, raw_data_path):
        """
        CORRECTED VERSION: Feature engineering happens BEFORE train/test split!

        This prevents data leakage by ensuring:
        1. Feature engineering runs once on full dataset
        2. Team statistics are cumulative across all data
        3. Train and test share the same feature engineering context
        4. Split happens AFTER features are created
        """
        try:
            logging.info('='*70)
            logging.info('DATA TRANSFORMATION STARTED')
            logging.info('='*70)

            # ================================================================
            # STEP 1: Read Raw Data
            # ================================================================
            logging.info('-'*70)
            logging.info('STEP 1: Loading Raw Data')
            logging.info('-'*70)
            logging.info(f'Reading raw data from: {raw_data_path}')

            df = pd.read_csv(raw_data_path)
            logging.info(
                f'✓ Raw data loaded: {df.shape[0]} rows, {df.shape[1]} columns')

            # Validate
            required_columns = ['Date', 'HomeTeam',
                                'AwayTeam', 'FTHG', 'FTAG', 'FTR']
            missing = set(required_columns) - set(df.columns)
            if missing:
                logging.error(f'Missing required columns: {missing}')
                raise ValueError(f'Missing columns: {missing}')
            logging.info('✓ All required columns present')

            # ================================================================
            # STEP 2: Feature Engineering on FULL Dataset
            # ================================================================
            logging.info('-'*70)
            logging.info('STEP 2: Feature Engineering (FULL DATASET)')
            logging.info('-'*70)
            logging.info(
                '⚠ CRITICAL: Processing entire dataset to avoid data leakage')

            df_features = self.process_football_features(df)
            logging.info(
                f'✓ Features created for full dataset: {df_features.shape}')

            # ================================================================
            # STEP 3: Train/Test Split (AFTER Feature Engineering)
            # ================================================================
            logging.info('-'*70)
            logging.info('STEP 3: Train/Test Split')
            logging.info('-'*70)
            logging.info('Splitting data (80/20, time-based, no shuffle)')

            # Calculate split index (80% train, 20% test)
            split_idx = int(len(df_features) * 0.8)

            train_df = df_features.iloc[:split_idx].reset_index(drop=True)
            test_df = df_features.iloc[split_idx:].reset_index(drop=True)

            logging.info(f'✓ Split completed:')
            logging.info(
                f'  Train: {len(train_df)} samples ({len(train_df)/len(df_features)*100:.1f}%)')
            logging.info(
                f'  Test:  {len(test_df)} samples ({len(test_df)/len(df_features)*100:.1f}%)')

            # Save the splits for reference
            train_df.to_csv(
                self.data_transformation_config.train_data_path, index=False)
            test_df.to_csv(
                self.data_transformation_config.test_data_path, index=False)
            logging.info(
                f'✓ Train data saved to: {self.data_transformation_config.train_data_path}')
            logging.info(
                f'✓ Test data saved to: {self.data_transformation_config.test_data_path}')

            # ================================================================
            # STEP 4: Separate Features and Target
            # ================================================================
            logging.info('-'*70)
            logging.info('STEP 4: Separating Features and Target')
            logging.info('-'*70)

            target_column_name = 'Result'
            target_mapping = {'H': 0, 'D': 1, 'A': 2}
            logging.info(f'Target column: {target_column_name}')
            logging.info(f'Target mapping: {target_mapping}')

            # Train
            input_feature_train_df = train_df.drop(
                columns=[target_column_name])
            target_feature_train_df = train_df[target_column_name].map(
                target_mapping)

            logging.info(
                f'Train features shape: {input_feature_train_df.shape}')
            logging.info(
                f'Train target shape: {target_feature_train_df.shape}')

            train_target_dist = target_feature_train_df.value_counts().sort_index()
            logging.info(f'Train target distribution:')
            for idx, count in train_target_dist.items():
                label = ['Home Win', 'Draw', 'Away Win'][idx]
                pct = count / len(target_feature_train_df) * 100
                logging.info(f'  {label} ({idx}): {count} ({pct:.1f}%)')

            # Test
            input_feature_test_df = test_df.drop(columns=[target_column_name])
            target_feature_test_df = test_df[target_column_name].map(
                target_mapping)

            logging.info(f'Test features shape: {input_feature_test_df.shape}')
            logging.info(f'Test target shape: {target_feature_test_df.shape}')

            test_target_dist = target_feature_test_df.value_counts().sort_index()
            logging.info(f'Test target distribution:')
            for idx, count in test_target_dist.items():
                label = ['Home Win', 'Draw', 'Away Win'][idx]
                pct = count / len(target_feature_test_df) * 100
                logging.info(f'  {label} ({idx}): {count} ({pct:.1f}%)')

            # Check for unmapped values
            if target_feature_train_df.isnull().any():
                invalid = train_df[target_column_name][target_feature_train_df.isnull(
                )].unique()
                logging.error(f'Invalid target values in train: {invalid}')
                raise ValueError(f'Invalid target values: {invalid}')

            if target_feature_test_df.isnull().any():
                invalid = test_df[target_column_name][target_feature_test_df.isnull(
                )].unique()
                logging.error(f'Invalid target values in test: {invalid}')
                raise ValueError(f'Invalid target values: {invalid}')

            logging.info('✓ No invalid target values')

            # ================================================================
            # STEP 5: Apply Preprocessing Pipeline
            # ================================================================
            logging.info('-'*70)
            logging.info('STEP 5: Preprocessing Pipeline')
            logging.info('-'*70)

            preprocessing_obj = self.get_data_transformer_object()

            # Fit on training data
            logging.info('Fitting preprocessing pipeline on training data...')
            try:
                preprocessing_obj.fit(
                    input_feature_train_df, target_feature_train_df)
                logging.info('✓ Pipeline fitted successfully')
            except Exception as e:
                logging.error(f'Pipeline fitting failed: {str(e)}')
                raise

            # Verify all transformers are fitted
            for step_name, transformer in preprocessing_obj.named_steps.items():
                if hasattr(transformer, 'is_fitted_'):
                    logging.info(
                        f'✓ {step_name}: is_fitted = {transformer.is_fitted_}')
                elif hasattr(transformer, 'n_features_in_'):
                    logging.info(
                        f'✓ {step_name}: n_features_in = {transformer.n_features_in_}')
                else:
                    logging.warning(
                        f'⚠ {step_name}: No fitted attributes found')

            # Transform training data
            logging.info('Transforming training data...')
            try:
                input_feature_train_arr = preprocessing_obj.transform(
                    input_feature_train_df)
                logging.info(
                    f'✓ Train data transformed: {input_feature_train_arr.shape}')
            except Exception as e:
                logging.error(f'Train transformation failed: {str(e)}')
                raise

            # Transform test data
            logging.info('Transforming test data...')
            try:
                input_feature_test_arr = preprocessing_obj.transform(
                    input_feature_test_df)
                logging.info(
                    f'✓ Test data transformed: {input_feature_test_arr.shape}')
            except Exception as e:
                logging.error(f'Test transformation failed: {str(e)}')
                raise

            # ================================================================
            # STEP 6: Combine Features and Target
            # ================================================================
            logging.info('-'*70)
            logging.info('STEP 6: Creating Final Arrays')
            logging.info('-'*70)

            train_arr = np.c_[
                input_feature_train_arr,
                np.array(target_feature_train_df)
            ]
            test_arr = np.c_[
                input_feature_test_arr,
                np.array(target_feature_test_df)
            ]

            logging.info(f'✓ Final train array: {train_arr.shape}')
            logging.info(f'✓ Final test array: {test_arr.shape}')

            # Verify no NaN values
            train_nan = np.isnan(train_arr).sum()
            test_nan = np.isnan(test_arr).sum()

            if train_nan > 0:
                logging.warning(
                    f'⚠ Train array contains {train_nan} NaN values')
            else:
                logging.info('✓ Train array has no NaN values')

            if test_nan > 0:
                logging.warning(f'⚠ Test array contains {test_nan} NaN values')
            else:
                logging.info('✓ Test array has no NaN values')

            # ================================================================
            # STEP 7: Save Preprocessing Object
            # ================================================================
            logging.info('-'*70)
            logging.info('STEP 7: Saving Artifacts')
            logging.info('-'*70)
            logging.info(
                f'Saving preprocessor to: {self.data_transformation_config.preprocessor_obj_file_path}')

            try:
                save_object(
                    file_path=self.data_transformation_config.preprocessor_obj_file_path,
                    obj=preprocessing_obj
                )
                logging.info('✓ Preprocessing object saved successfully')
            except Exception as e:
                logging.error(f'Failed to save preprocessor: {str(e)}')
                raise

            # ================================================================
            # COMPLETION SUMMARY
            # ================================================================
            logging.info('='*70)
            logging.info('DATA TRANSFORMATION COMPLETED SUCCESSFULLY')
            logging.info('='*70)
            logging.info('Summary:')
            logging.info(f'  Original raw data: {len(df)} matches')
            logging.info(
                f'  After feature engineering: {len(df_features)} samples')
            logging.info(
                f'  Train samples: {train_arr.shape[0]} ({train_arr.shape[0]/len(df_features)*100:.1f}%)')
            logging.info(
                f'  Test samples: {test_arr.shape[0]} ({test_arr.shape[0]/len(df_features)*100:.1f}%)')
            logging.info(f'  Features per sample: {train_arr.shape[1] - 1}')
            logging.info(f'  Files created:')
            logging.info(
                f'    - {self.data_transformation_config.train_data_path}')
            logging.info(
                f'    - {self.data_transformation_config.test_data_path}')
            logging.info(
                f'    - {self.data_transformation_config.preprocessor_obj_file_path}')
            logging.info('='*70)

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            logging.error('='*70)
            logging.error('DATA TRANSFORMATION FAILED')
            logging.error('='*70)
            logging.error(f'Error: {str(e)}')
            logging.error('='*70)
            raise CustomException(e, sys)


if __name__ == "__main__":
    from src.components.data_ingestion import DataIngestion

    # Step 1: Data Ingestion
    logging.info('\n' + '='*70)
    logging.info('RUNNING CORRECTED PIPELINE')
    logging.info('='*70 + '\n')

    obj = DataIngestion()
    raw_data_path = obj.initiate_data_ingestion()

    # Step 2: Data Transformation
    data_transformation = DataTransformation()
    train_arr, test_arr, preprocessor_path = data_transformation.initiate_data_transformation(
        raw_data_path
    )

    print(f"\n{'='*70}")
    print("✓ PIPELINE COMPLETED SUCCESSFULLY")
    print(f"{'='*70}")
    print(f"Train array shape: {train_arr.shape}")
    print(f"Test array shape: {test_arr.shape}")
    print(f"Preprocessor saved: {preprocessor_path}")
    print(f"{'='*70}\n")
