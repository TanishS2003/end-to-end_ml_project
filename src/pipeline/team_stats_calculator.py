"""
Team Statistics Calculator
Calculates current team statistics from historical data for realistic predictions
"""

import sys
import pandas as pd
import os
from typing import Dict, Tuple
from src.logger import logging
from src.exception import CustomException


class TeamStatsCalculator:
    """
    Calculate current team statistics from historical match data
    """

    def __init__(self, data_path: str = 'artifacts/data.csv'):
        """
        Initialize with path to raw historical data

        Args:
            data_path: Path to CSV file with historical matches
        """
        try:
            logging.info('='*70)
            logging.info('Initializing TeamStatsCalculator')
            logging.info('='*70)
            logging.info(f'Data path: {data_path}')

            self.data_path = data_path
            self.team_stats = {}

            if os.path.exists(data_path):
                logging.info(f'Historical data file found: {data_path}')
                self._calculate_stats()
            else:
                logging.warning(f'Historical data file not found: {data_path}')
                logging.warning('Team statistics will not be available')

        except Exception as e:
            logging.error(f'Error initializing TeamStatsCalculator: {str(e)}')
            raise CustomException(e, sys)

    def _calculate_stats(self):
        """
        Calculate current statistics for all teams from historical data
        """
        try:
            logging.info('Calculating team statistics from historical data')

            # Read historical data
            df = pd.read_csv(self.data_path)
            logging.info(f'Loaded {len(df)} matches from historical data')

            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values('Date')
            logging.info('Data sorted by date')

            # Get unique teams
            teams = pd.concat([df['HomeTeam'], df['AwayTeam']]).unique()
            logging.info(f'Found {len(teams)} unique teams')

            # Initialize stats for each team
            for team in teams:
                self.team_stats[team] = {
                    'MP': 0, 'Pts': 0, 'Gls': 0, 'GA': 0,
                    'H_MP': 0, 'H_Pts': 0, 'A_MP': 0, 'A_Pts': 0
                }

            logging.info('Processing matches to calculate running statistics')

            # Calculate running stats
            for idx, row in df.iterrows():
                home_t = row['HomeTeam']
                away_t = row['AwayTeam']

                # Update Home Team stats
                h_pts = 3 if row['FTR'] == 'H' else (
                    1 if row['FTR'] == 'D' else 0)
                self.team_stats[home_t]['MP'] += 1
                self.team_stats[home_t]['Pts'] += h_pts
                self.team_stats[home_t]['Gls'] += row['FTHG']
                self.team_stats[home_t]['GA'] += row['FTAG']
                self.team_stats[home_t]['H_MP'] += 1
                self.team_stats[home_t]['H_Pts'] += h_pts

                # Update Away Team stats
                a_pts = 3 if row['FTR'] == 'A' else (
                    1 if row['FTR'] == 'D' else 0)
                self.team_stats[away_t]['MP'] += 1
                self.team_stats[away_t]['Pts'] += a_pts
                self.team_stats[away_t]['Gls'] += row['FTAG']
                self.team_stats[away_t]['GA'] += row['FTHG']
                self.team_stats[away_t]['A_MP'] += 1
                self.team_stats[away_t]['A_Pts'] += a_pts

            logging.info('✓ Running statistics calculated for all teams')

            # Calculate rankings
            self._calculate_rankings()

            logging.info('='*70)
            logging.info('Team statistics calculation completed')
            logging.info('='*70)

        except FileNotFoundError as e:
            logging.error(f'Data file not found: {self.data_path}')
            raise CustomException(e, sys)
        except KeyError as e:
            logging.error(f'Missing required column in data: {str(e)}')
            raise CustomException(e, sys)
        except Exception as e:
            logging.error(f'Error calculating team statistics: {str(e)}')
            raise CustomException(e, sys)

    def _calculate_rankings(self):
        """
        Calculate current league rankings based on points, GD, and goals
        """
        try:
            logging.info('Calculating team rankings')

            # Create table for ranking
            table = []
            for team, stats in self.team_stats.items():
                if stats['MP'] > 0:  # Only rank teams that have played
                    table.append({
                        'Team': team,
                        'Pts': stats['Pts'],
                        'GD': stats['Gls'] - stats['GA'],
                        'Gls': stats['Gls']
                    })

            # Sort by points, then GD, then goals
            rank_df = pd.DataFrame(table).sort_values(
                by=['Pts', 'GD', 'Gls'], ascending=False
            ).reset_index(drop=True)

            # Add rankings to team stats
            for idx, row in rank_df.iterrows():
                team = row['Team']
                self.team_stats[team]['Rank'] = idx + 1

            logging.info(f'✓ Rankings calculated for {len(table)} teams')

        except Exception as e:
            logging.error(f'Error calculating rankings: {str(e)}')
            raise CustomException(e, sys)

    def get_team_stats(self, team_name: str) -> Dict:
        """
        Get statistics for a specific team

        Args:
            team_name: Name of the team

        Returns:
            Dictionary with team statistics
        """
        try:
            logging.info(f'Getting stats for team: {team_name}')

            if team_name not in self.team_stats:
                logging.warning(f'Team not found in stats: {team_name}')
                logging.warning('Returning default statistics')
                # Return default stats if team not found
                return {
                    'Rank': 10,
                    'Pts_MP': 1.5,
                    'GD': 0,
                    'H_Pts_MP': 1.5,
                    'A_Pts_MP': 1.5,
                    'Gls': 0,
                    'GA': 0
                }

            stats = self.team_stats[team_name]
            mp = stats['MP']
            h_mp = stats['H_MP']
            a_mp = stats['A_MP']

            calculated_stats = {
                'Rank': stats.get('Rank', 10),
                'Pts_MP': stats['Pts'] / mp if mp > 0 else 0,
                'GD': stats['Gls'] - stats['GA'],
                'H_Pts_MP': stats['H_Pts'] / h_mp if h_mp > 0 else 0,
                'A_Pts_MP': stats['A_Pts'] / a_mp if a_mp > 0 else 0,
                'Gls': stats['Gls'],
                'GA': stats['GA']
            }

            logging.info(
                f'✓ Stats retrieved for {team_name}: Rank={calculated_stats["Rank"]}, Pts/MP={calculated_stats["Pts_MP"]:.2f}')

            return calculated_stats

        except Exception as e:
            logging.error(
                f'Error getting team stats for {team_name}: {str(e)}')
            raise CustomException(e, sys)

    def get_match_features(self, home_team: str, away_team: str) -> Tuple[Dict, Dict]:
        """
        Get features for both teams in a match

        Args:
            home_team: Name of home team
            away_team: Name of away team

        Returns:
            Tuple of (home_stats, away_stats)
        """
        try:
            logging.info(f'Getting match features: {home_team} vs {away_team}')

            home_stats = self.get_team_stats(home_team)
            away_stats = self.get_team_stats(away_team)

            logging.info(f'✓ Match features retrieved')

            return home_stats, away_stats

        except Exception as e:
            logging.error(f'Error getting match features: {str(e)}')
            raise CustomException(e, sys)

    def get_league_table(self, top_n: int = None) -> pd.DataFrame:
        """
        Get current league table

        Args:
            top_n: Return only top N teams (None for all)

        Returns:
            DataFrame with league standings
        """
        try:
            logging.info(
                f'Generating league table (top {top_n if top_n else "all"})')

            table_data = []

            for team, stats in self.team_stats.items():
                if stats['MP'] > 0:
                    table_data.append({
                        'Rank': stats.get('Rank', '-'),
                        'Team': team,
                        'MP': stats['MP'],
                        'Pts': stats['Pts'],
                        'GD': stats['Gls'] - stats['GA'],
                        'Gls': stats['Gls'],
                        'GA': stats['GA'],
                        'Pts/MP': round(stats['Pts'] / stats['MP'], 2) if stats['MP'] > 0 else 0
                    })

            df = pd.DataFrame(table_data).sort_values('Rank')

            if top_n:
                df = df.head(top_n)

            logging.info(f'✓ League table generated with {len(df)} teams')

            return df

        except Exception as e:
            logging.error(f'Error generating league table: {str(e)}')
            raise CustomException(e, sys)

    def get_team_form(self, team_name: str, n_matches: int = 5) -> str:
        """
        Get recent form for a team (last N matches)

        Args:
            team_name: Name of the team
            n_matches: Number of recent matches to check

        Returns:
            String like "W-W-D-L-W" representing recent form
        """
        # This is a simplified version - you'd need to track match history
        # For now, return placeholder
        logging.info(f'get_team_form called for {team_name} (not implemented)')
        return "N/A"


# Convenience function
def get_team_stats_for_prediction(home_team: str, away_team: str) -> Tuple[Dict, Dict]:
    """
    Quick function to get team stats for prediction

    Args:
        home_team: Home team name
        away_team: Away team name

    Returns:
        Tuple of (home_stats, away_stats)
    """
    try:
        logging.info(
            f'get_team_stats_for_prediction: {home_team} vs {away_team}')

        calculator = TeamStatsCalculator()
        return calculator.get_match_features(home_team, away_team)

    except Exception as e:
        logging.error(f'Error in get_team_stats_for_prediction: {str(e)}')
        raise CustomException(e, sys)


if __name__ == "__main__":
    # Test the calculator
    try:
        logging.info("="*70)
        logging.info("TEAM STATISTICS CALCULATOR TEST")
        logging.info("="*70)

        print("\n" + "="*70)
        print("TEAM STATISTICS CALCULATOR TEST")
        print("="*70 + "\n")

        calculator = TeamStatsCalculator()

        # Test match
        home_team = "Barcelona"
        away_team = "Real Madrid"

        print(f"Match: {home_team} vs {away_team}\n")

        home_stats, away_stats = calculator.get_match_features(
            home_team, away_team)

        print(f"{home_team} Stats:")
        for key, value in home_stats.items():
            print(f"  {key}: {value}")

        print(f"\n{away_team} Stats:")
        for key, value in away_stats.items():
            print(f"  {key}: {value}")

        print("\n" + "="*70)
        print("LEAGUE TABLE (Top 10)")
        print("="*70 + "\n")

        table = calculator.get_league_table(top_n=10)
        print(table.to_string(index=False))

        print("\n" + "="*70 + "\n")

        logging.info("="*70)
        logging.info("Test completed successfully")
        logging.info("="*70)

    except CustomException as e:
        print(f"CustomException: {e}")
        logging.error(f"Test failed: {e}")
    except Exception as e:
        print(f"Error: {e}")
        logging.error(f"Unexpected error: {e}")
    """
    Calculate current team statistics from historical match data
    """

    def __init__(self, data_path: str = 'artifacts/data.csv'):
        """
        Initialize with path to raw historical data

        Args:
            data_path: Path to CSV file with historical matches
        """
        self.data_path = data_path
        self.team_stats = {}

        if os.path.exists(data_path):
            self._calculate_stats()

    def _calculate_stats(self):
        """
        Calculate current statistics for all teams from historical data
        """
        # Read historical data
        df = pd.read_csv(self.data_path)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')

        # Get unique teams
        teams = pd.concat([df['HomeTeam'], df['AwayTeam']]).unique()

        # Initialize stats for each team
        for team in teams:
            self.team_stats[team] = {
                'MP': 0, 'Pts': 0, 'Gls': 0, 'GA': 0,
                'H_MP': 0, 'H_Pts': 0, 'A_MP': 0, 'A_Pts': 0
            }

        # Calculate running stats
        for _, row in df.iterrows():
            home_t = row['HomeTeam']
            away_t = row['AwayTeam']

            # Update Home Team stats
            h_pts = 3 if row['FTR'] == 'H' else (1 if row['FTR'] == 'D' else 0)
            self.team_stats[home_t]['MP'] += 1
            self.team_stats[home_t]['Pts'] += h_pts
            self.team_stats[home_t]['Gls'] += row['FTHG']
            self.team_stats[home_t]['GA'] += row['FTAG']
            self.team_stats[home_t]['H_MP'] += 1
            self.team_stats[home_t]['H_Pts'] += h_pts

            # Update Away Team stats
            a_pts = 3 if row['FTR'] == 'A' else (1 if row['FTR'] == 'D' else 0)
            self.team_stats[away_t]['MP'] += 1
            self.team_stats[away_t]['Pts'] += a_pts
            self.team_stats[away_t]['Gls'] += row['FTAG']
            self.team_stats[away_t]['GA'] += row['FTHG']
            self.team_stats[away_t]['A_MP'] += 1
            self.team_stats[away_t]['A_Pts'] += a_pts

        # Calculate rankings
        self._calculate_rankings()

    def _calculate_rankings(self):
        """
        Calculate current league rankings based on points, GD, and goals
        """
        # Create table for ranking
        table = []
        for team, stats in self.team_stats.items():
            if stats['MP'] > 0:  # Only rank teams that have played
                table.append({
                    'Team': team,
                    'Pts': stats['Pts'],
                    'GD': stats['Gls'] - stats['GA'],
                    'Gls': stats['Gls']
                })

        # Sort by points, then GD, then goals
        rank_df = pd.DataFrame(table).sort_values(
            by=['Pts', 'GD', 'Gls'], ascending=False
        ).reset_index(drop=True)

        # Add rankings to team stats
        for idx, row in rank_df.iterrows():
            team = row['Team']
            self.team_stats[team]['Rank'] = idx + 1

    def get_team_stats(self, team_name: str) -> Dict:
        """
        Get statistics for a specific team

        Args:
            team_name: Name of the team

        Returns:
            Dictionary with team statistics
        """
        if team_name not in self.team_stats:
            # Return default stats if team not found
            return {
                'Rank': 10,
                'Pts_MP': 1.5,
                'GD': 0,
                'H_Pts_MP': 1.5,
                'A_Pts_MP': 1.5,
                'Gls': 0,
                'GA': 0
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

    def get_match_features(self, home_team: str, away_team: str) -> Tuple[Dict, Dict]:
        """
        Get features for both teams in a match

        Args:
            home_team: Name of home team
            away_team: Name of away team

        Returns:
            Tuple of (home_stats, away_stats)
        """
        home_stats = self.get_team_stats(home_team)
        away_stats = self.get_team_stats(away_team)

        return home_stats, away_stats

    def get_league_table(self, top_n: int = None) -> pd.DataFrame:
        """
        Get current league table

        Args:
            top_n: Return only top N teams (None for all)

        Returns:
            DataFrame with league standings
        """
        table_data = []

        for team, stats in self.team_stats.items():
            if stats['MP'] > 0:
                table_data.append({
                    'Rank': stats.get('Rank', '-'),
                    'Team': team,
                    'MP': stats['MP'],
                    'Pts': stats['Pts'],
                    'GD': stats['Gls'] - stats['GA'],
                    'Gls': stats['Gls'],
                    'GA': stats['GA'],
                    'Pts/MP': round(stats['Pts'] / stats['MP'], 2) if stats['MP'] > 0 else 0
                })

        df = pd.DataFrame(table_data).sort_values('Rank')

        if top_n:
            return df.head(top_n)

        return df

    def get_team_form(self, team_name: str, n_matches: int = 5) -> str:
        """
        Get recent form for a team (last N matches)

        Args:
            team_name: Name of the team
            n_matches: Number of recent matches to check

        Returns:
            String like "W-W-D-L-W" representing recent form
        """
        # This is a simplified version - you'd need to track match history
        # For now, return placeholder
        return "N/A"


# Convenience function
def get_team_stats_for_prediction(home_team: str, away_team: str) -> Tuple[Dict, Dict]:
    """
    Quick function to get team stats for prediction

    Args:
        home_team: Home team name
        away_team: Away team name

    Returns:
        Tuple of (home_stats, away_stats)
    """
    calculator = TeamStatsCalculator()
    return calculator.get_match_features(home_team, away_team)


if __name__ == "__main__":
    # Test the calculator
    print("\n" + "="*70)
    print("TEAM STATISTICS CALCULATOR TEST")
    print("="*70 + "\n")

    calculator = TeamStatsCalculator()

    # Test match
    home_team = "Barcelona"
    away_team = "Real Madrid"

    print(f"Match: {home_team} vs {away_team}\n")

    home_stats, away_stats = calculator.get_match_features(
        home_team, away_team)

    print(f"{home_team} Stats:")
    for key, value in home_stats.items():
        print(f"  {key}: {value}")

    print(f"\n{away_team} Stats:")
    for key, value in away_stats.items():
        print(f"  {key}: {value}")

    print("\n" + "="*70)
    print("LEAGUE TABLE (Top 10)")
    print("="*70 + "\n")

    table = calculator.get_league_table(top_n=10)
    print(table.to_string(index=False))

    print("\n" + "="*70 + "\n")
