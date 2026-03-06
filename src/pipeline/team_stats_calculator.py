"""
Team Statistics Calculator - Updated for Advanced Underlying Metrics
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
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df = df.sort_values('Date')

            teams = pd.concat([df['HomeTeam'], df['AwayTeam']]).unique()

            for team in teams:
                self.team_stats[team] = {
                    'MP': 0, 'Pts': 0, 'Gls': 0, 'GA': 0,
                    'H_MP': 0, 'H_Pts': 0, 'A_MP': 0, 'A_Pts': 0,
                    # --- NEW TRACKERS ---
                    'SoT': 0, 'SoT_Conceded': 0, 'Corners': 0, 'Cards': 0,
                    'last_5_results': []
                }

            for _, row in df.iterrows():
                home_t = row['HomeTeam']
                away_t = row['AwayTeam']

                h_pts = 3 if row['FTR'] == 'H' else (
                    1 if row['FTR'] == 'D' else 0)
                a_pts = 3 if row['FTR'] == 'A' else (
                    1 if row['FTR'] == 'D' else 0)

                # Extract Underlying Match Stats Safely
                h_sot = row['HST'] if pd.notna(row['HST']) else 0
                a_sot = row['AST'] if pd.notna(row['AST']) else 0
                h_cor = row['HC'] if pd.notna(row['HC']) else 0
                a_cor = row['AC'] if pd.notna(row['AC']) else 0

                h_cards = (row['HY'] if pd.notna(row['HY']) else 0) + \
                    (row['HR'] if pd.notna(row['HR']) else 0)
                a_cards = (row['AY'] if pd.notna(row['AY']) else 0) + \
                    (row['AR'] if pd.notna(row['AR']) else 0)

                # Update Home Stats
                self.team_stats[home_t]['MP'] += 1
                self.team_stats[home_t]['Pts'] += h_pts
                self.team_stats[home_t]['Gls'] += row['FTHG']
                self.team_stats[home_t]['GA'] += row['FTAG']
                self.team_stats[home_t]['H_MP'] += 1
                self.team_stats[home_t]['H_Pts'] += h_pts
                self.team_stats[home_t]['SoT'] += h_sot
                self.team_stats[home_t]['SoT_Conceded'] += a_sot
                self.team_stats[home_t]['Corners'] += h_cor
                self.team_stats[home_t]['Cards'] += h_cards

                # Update Home Momentum (Now includes SoT)
                self.team_stats[home_t]['last_5_results'].append(
                    {'pts': h_pts, 'gf': row['FTHG'],
                        'ga': row['FTAG'], 'sot': h_sot, 'sot_c': a_sot}
                )
                if len(self.team_stats[home_t]['last_5_results']) > 5:
                    self.team_stats[home_t]['last_5_results'].pop(0)

                # Update Away Stats
                self.team_stats[away_t]['MP'] += 1
                self.team_stats[away_t]['Pts'] += a_pts
                self.team_stats[away_t]['Gls'] += row['FTAG']
                self.team_stats[away_t]['GA'] += row['FTHG']
                self.team_stats[away_t]['A_MP'] += 1
                self.team_stats[away_t]['A_Pts'] += a_pts
                self.team_stats[away_t]['SoT'] += a_sot
                self.team_stats[away_t]['SoT_Conceded'] += h_sot
                self.team_stats[away_t]['Corners'] += a_cor
                self.team_stats[away_t]['Cards'] += a_cards

                # Update Away Momentum (Now includes SoT)
                self.team_stats[away_t]['last_5_results'].append(
                    {'pts': a_pts, 'gf': row['FTAG'],
                        'ga': row['FTHG'], 'sot': a_sot, 'sot_c': h_sot}
                )
                if len(self.team_stats[away_t]['last_5_results']) > 5:
                    self.team_stats[away_t]['last_5_results'].pop(0)

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
        if team_name not in self.team_stats:
            logging.warning(f"Team {team_name} not found in historical data.")
            return {
                'Rank': 10, 'Pts_MP': 1.5, 'GD': 0, 'H_Pts_MP': 1.5,
                'A_Pts_MP': 1.5, 'Gls': 0, 'GA': 0, 'MP': 1,
                'Form_Pts': 7, 'Form_GF': 2, 'Form_GA': 1,
                'SoT_MP': 4.0, 'SoT_Conceded_MP': 4.0, 'Corners_MP': 4.5,
                'Cards_MP': 2.0, 'Conversion': 0.3,
                'Form_SoT': 20, 'Form_SoT_C': 20
            }

        stats = self.team_stats[team_name]
        mp = max(stats['MP'], 1)
        h_mp = max(stats['H_MP'], 1)
        a_mp = max(stats['A_MP'], 1)
        last_5 = stats.get('last_5_results', [])

        return {
            'Rank': stats.get('Rank', 10),
            'Pts_MP': stats['Pts'] / mp,
            'GD': (stats['Gls'] - stats['GA']) / mp,
            'Gls': stats['Gls'] / mp,
            'GA': stats['GA'] / mp,
            'H_Pts_MP': stats['H_Pts'] / h_mp,
            'A_Pts_MP': stats['A_Pts'] / a_mp,
            'MP': mp,

            # --- NEW UNDERLYING METRICS ---
            'SoT_MP': stats['SoT'] / mp,
            'SoT_Conceded_MP': stats['SoT_Conceded'] / mp,
            'Corners_MP': stats['Corners'] / mp,
            'Cards_MP': stats['Cards'] / mp,
            'Conversion': stats['Gls'] / max(stats['SoT'], 1),

            # --- DYNAMIC FORM (Including SoT) ---
            'Form_Pts': sum([g['pts'] for g in last_5]),
            'Form_GF': sum([g['gf'] for g in last_5]),
            'Form_GA': sum([g['ga'] for g in last_5]),
            'Form_SoT': sum([g['sot'] for g in last_5]),
            'Form_SoT_C': sum([g['sot_c'] for g in last_5])
        }
