import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from dataclasses import dataclass


@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join('artifacts', 'data.csv')


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
        logging.info('DataIngestion initialized')
        logging.info(
            f'Raw data will be saved to: {self.ingestion_config.raw_data_path}')

    def initiate_data_ingestion(self):
        logging.info('='*70)
        logging.info('DATA INGESTION STARTED')
        logging.info('='*70)

        try:
            # List of CSV files (Past 3 seasons + current)
            csv_files = [
                'notebook/SP1 (3).csv',
                'notebook/SP1 (2).csv',
                'notebook/SP1 (1).csv',
                'notebook/SP1.csv'
            ]

            logging.info(f'Files to process: {len(csv_files)}')

            # Read and concatenate all CSV files
            logging.info('Reading dataset from multiple files')
            dfs = []
            for i, file in enumerate(csv_files, 1):
                logging.info(f'Reading file {i}/{len(csv_files)}: {file}')
                df = pd.read_csv(file)
                logging.info(f'✓ Loaded {len(df)} rows from {file}')
                dfs.append(df)

            # Concatenate all dataframes
            df = pd.concat(dfs, ignore_index=True)
            logging.info(
                f'✓ Concatenated all files: {len(df)} total rows, {len(df.columns)} columns')

            # Validate required columns
            required_columns = ['Date', 'HomeTeam',
                                'AwayTeam', 'FTHG', 'FTAG', 'FTR']
            missing_cols = set(required_columns) - set(df.columns)
            if missing_cols:
                logging.error(f'Missing required columns: {missing_cols}')
                raise ValueError(f'Missing required columns: {missing_cols}')
            logging.info('✓ All required columns present')

            # Check for null values
            null_counts = df[required_columns].isnull().sum()
            total_nulls = null_counts.sum()
            if total_nulls > 0:
                logging.warning(f'Found {total_nulls} null values:')
                for col, count in null_counts[null_counts > 0].items():
                    logging.warning(f'  {col}: {count} nulls')
            else:
                logging.info('✓ No null values in required columns')

            # Sort by date to ensure chronological order
            logging.info('Sorting data by date...')
            df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
            df = df.sort_values('Date').reset_index(drop=True)
            logging.info(f'✓ Data sorted chronologically')
            logging.info(
                f'  Date range: {df["Date"].min()} to {df["Date"].max()}')

            # Create artifacts directory
            os.makedirs(os.path.dirname(
                self.ingestion_config.raw_data_path), exist_ok=True)

            # Save raw data
            logging.info(
                f'Saving raw data to {self.ingestion_config.raw_data_path}')
            df.to_csv(self.ingestion_config.raw_data_path,
                      index=False, header=True)
            logging.info('✓ Raw data saved successfully')

            logging.info('='*70)
            logging.info('DATA INGESTION COMPLETED')
            logging.info('='*70)
            logging.info(f'Summary:')
            logging.info(f'  - Total matches: {len(df)}')
            logging.info(f'  - Columns: {len(df.columns)}')
            logging.info(
                f'  - Date range: {df["Date"].min().date()} to {df["Date"].max().date()}')
            logging.info(
                f'  - Unique teams: {pd.concat([df["HomeTeam"], df["AwayTeam"]]).nunique()}')
            logging.info(
                f'  - Raw data saved: {self.ingestion_config.raw_data_path}')
            logging.info('='*70)

            return self.ingestion_config.raw_data_path

        except Exception as e:
            logging.error('='*70)
            logging.error('DATA INGESTION FAILED')
            logging.error('='*70)
            logging.error(f'Error: {str(e)}')
            logging.error('='*70)
            raise CustomException(e, sys)


if __name__ == "__main__":
    obj = DataIngestion()
    raw_data_path = obj.initiate_data_ingestion()
    print(f"\n✓ Raw data saved to: {raw_data_path}\n")
