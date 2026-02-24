"""
Team Statistics Calculator
"""

import sys
import pandas as pd
import os
from src.logger import logging
from src.exception import CustomException


class TeamStatsCalculator:
    def __init__(self, data_path: str = 'artifacts/data.csv'):
        try:
            logging.info('TeamStatsCalculator initialized')
            self.data_path = data_path
            self.team_stats = {}

            if os.path.exists(data_path):
                self._calculate_stats()
            else:
                logging.warning(f"Data file not found at {data_path}")

        except Exception as e:
            raise CustomException(e, sys)

    def _calculate_stats(self):
        try:
            df = pd.read_csv(self.data_path)
            # Ensure Date is handled correctly for sorting
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df = df.sort_values('Date')

            teams = pd.concat([df['HomeTeam'], df['AwayTeam']]).unique()

            for team in teams:
                self.team_stats[team] = {
                    'MP': 0, 'Pts': 0, 'Gls': 0, 'GA': 0,
                    'H_MP': 0, 'H_Pts': 0, 'A_MP': 0, 'A_Pts': 0
                }

            for _, row in df.iterrows():
                home_t = row['HomeTeam']
                away_t = row['AwayTeam']

                h_pts = 3 if row['FTR'] == 'H' else (
                    1 if row['FTR'] == 'D' else 0)
                a_pts = 3 if row['FTR'] == 'A' else (
                    1 if row['FTR'] == 'D' else 0)

                # Update Home Stats
                self.team_stats[home_t]['MP'] += 1
                self.team_stats[home_t]['Pts'] += h_pts
                self.team_stats[home_t]['Gls'] += row['FTHG']
                self.team_stats[home_t]['GA'] += row['FTAG']
                self.team_stats[home_t]['H_MP'] += 1
                self.team_stats[home_t]['H_Pts'] += h_pts

                # Update Away Stats
                self.team_stats[away_t]['MP'] += 1
                self.team_stats[away_t]['Pts'] += a_pts
                self.team_stats[away_t]['Gls'] += row['FTAG']
                self.team_stats[away_t]['GA'] += row['FTHG']
                self.team_stats[away_t]['A_MP'] += 1
                self.team_stats[away_t]['A_Pts'] += a_pts

            self._calculate_rankings()

        except Exception as e:
            raise CustomException(e, sys)

    def _calculate_rankings(self):
        table = []
        for team, stats in self.team_stats.items():
            if stats['MP'] > 0:
                table.append({
                    'Team': team,
                    'Pts': stats['Pts'],
                    'GD': stats['Gls'] - stats['GA'],
                    'Gls': stats['Gls']
                })

        if not table:
            return

        rank_df = pd.DataFrame(table).sort_values(
            by=['Pts', 'GD', 'Gls'], ascending=False
        ).reset_index(drop=True)

        for idx, row in rank_df.iterrows():
            self.team_stats[row['Team']]['Rank'] = idx + 1

    def get_team_stats(self, team_name: str):
        # Graceful fallback for unknown teams
        if team_name not in self.team_stats:
            logging.warning(f"Team {team_name} not found in historical data.")
            return {
                'Rank': 10, 'Pts_MP': 1.5, 'GD': 0,
                'H_Pts_MP': 1.5, 'A_Pts_MP': 1.5, 'Gls': 0, 'GA': 0
            }

        stats = self.team_stats[team_name]
        mp = stats['MP']
        h_mp = stats['H_MP']
        a_mp = stats['A_MP']

        return {
            'Rank': stats.get('Rank', 10),
            'Pts_MP': stats['Pts'] / mp if mp > 0 else 0,
            'GD': stats['Gls'] - stats['GA'],
            'H_Pts_MP': stats['H_Pts'] / h_mp if h_mp > 0 else 0,
            'A_Pts_MP': stats['A_Pts'] / a_mp if a_mp > 0 else 0,
            'Gls': stats['Gls'],
            'GA': stats['GA']
        }
