"""
Data Ingestion Component
Auto-downloads La Liga data from Football-Data.co.uk
"""

import os
import sys
import pandas as pd
import requests
from io import StringIO
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging


@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join('artifacts', 'data.csv')


class DataIngestion:
    def __init__(self):
        logging.info('DataIngestion initialized')
        self.ingestion_config = DataIngestionConfig()

        # 6 seasons of La Liga data
        self.dataset_urls = {
            '2021-22': 'https://www.football-data.co.uk/mmz4281/2122/SP1.csv',
            '2022-23': 'https://www.football-data.co.uk/mmz4281/2223/SP1.csv',
            '2023-24': 'https://www.football-data.co.uk/mmz4281/2324/SP1.csv',
            '2024-25': 'https://www.football-data.co.uk/mmz4281/2425/SP1.csv',
            '2025-26': 'https://www.football-data.co.uk/mmz4281/2526/SP1.csv',
        }

    def download_data_from_url(self, url: str, season: str) -> pd.DataFrame:
        try:
            logging.info(f'Downloading {season}...')
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            df = pd.read_csv(StringIO(response.text))
            df['Season'] = season
            logging.info(f'✓ {season}: {len(df)} matches')
            return df
        except Exception as e:
            logging.error(f'Error downloading {season}: {str(e)}')
            raise CustomException(e, sys)

    def initiate_data_ingestion(self):
        try:
            logging.info("="*70)
            logging.info("DATA INGESTION STARTED")
            logging.info("="*70)

            os.makedirs(os.path.dirname(
                self.ingestion_config.raw_data_path), exist_ok=True)

            all_dataframes = []
            for season, url in self.dataset_urls.items():
                df = self.download_data_from_url(url, season)
                all_dataframes.append(df)

            df = pd.concat(all_dataframes, ignore_index=True)
            logging.info(f'✓ Total matches: {len(df)}')

            # Validate
            required_columns = ['Date', 'HomeTeam',
                                'AwayTeam', 'FTHG', 'FTAG', 'FTR']
            missing = [c for c in required_columns if c not in df.columns]
            if missing:
                raise ValueError(f'Missing columns: {missing}')

            # Sort by date
            df['Date'] = pd.to_datetime(
                df['Date'], format='%d/%m/%Y', errors='coerce')
            df = df.sort_values('Date').reset_index(drop=True)

            df.to_csv(self.ingestion_config.raw_data_path, index=False)

            logging.info("DATA INGESTION COMPLETED")
            logging.info("="*70)

            return self.ingestion_config.raw_data_path

        except Exception as e:
            logging.error(f'Error: {str(e)}')
            raise CustomException(e, sys)
