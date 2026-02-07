import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass


@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'data.csv')


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info('Entered the data ingestion method or component')

        try:
            # List of CSV files (Past 3 seasons + current)
            csv_files = [
                'notebook/SP1 (3).csv',
                'notebook/SP1 (2).csv',
                'notebook/SP1 (1).csv',
                'notebook/SP1.csv'
            ]

            # Read and concatenate all CSV files
            logging.info('Reading the dataset from multiple files')
            dfs = []
            for file in csv_files:
                logging.info(f'Reading file: {file}')
                df = pd.read_csv(file)
                dfs.append(df)

            df = pd.concat(dfs, ignore_index=True)
            logging.info(f'Read dataset completed with {len(df)} total rows')

            # Create artifacts directory
            os.makedirs(os.path.dirname(
                self.ingestion_config.train_data_path), exist_ok=True)

            # Save raw data
            df.to_csv(self.ingestion_config.raw_data_path,
                      index=False, header=True)
            logging.info('Raw data saved to artifacts')

            # Train-test split (time-based split recommended for time series)
            logging.info('Train test split initiated')
            # Using shuffle=False to maintain temporal order for football data
            train_set, test_set = train_test_split(
                df, test_size=0.2, shuffle=False)

            train_set.to_csv(
                self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,
                            index=False, header=True)

            logging.info('Ingestion of the data is completed')

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()
