"""
Streamlit Web Application - Final Production Version
"""

from src.logger import logging
from src.exception import CustomException
from src.pipeline.predict_pipeline import PredictPipeline, CustomData
from src.pipeline.team_stats_calculator import TeamStatsCalculator
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os

# Page config
st.set_page_config(page_title="La Liga Predictor",
                   page_icon="⚽", layout="wide")

TEAMS = sorted([
    'Alaves', 'Almeria', 'Ath Bilbao', 'Ath Madrid', 'Barcelona',
    'Betis', 'Cadiz', 'Celta', 'Elche', 'Espanol', 'Getafe',
    'Girona', 'Granada', 'Las Palmas', 'Leganes', 'Levante',
    'Mallorca', 'Osasuna', 'Real Madrid', 'Real Sociedad',
    'Sevilla', 'Valencia', 'Vallecano', 'Villarreal'
])


def main():
    st.title("⚽ La Liga Match Predictor")
    st.markdown("---")

    # Sidebar Match Selection
    st.sidebar.header("Match Setup")
    home_team = st.sidebar.selectbox(
        "🏠 Home Team", TEAMS, index=TEAMS.index('Barcelona'))
    away_team = st.sidebar.selectbox(
        "🚀 Away Team", TEAMS, index=TEAMS.index('Real Madrid'))

    if home_team == away_team:
        st.sidebar.error("Please select two different teams.")
        return

    if not os.path.exists('artifacts/data.csv'):
        st.error("⚠️ Data file not found. Please run the training pipeline first.")
        return

    calc = TeamStatsCalculator()
    h_stats = calc.get_team_stats(home_team)
    a_stats = calc.get_team_stats(away_team)

    # Calculation logic for explanation
    h_quality = (h_stats['Pts_MP'] * 0.5) + (h_stats['GD']/20 * 0.5)
    a_quality = (a_stats['Pts_MP'] * 0.5) + (a_stats['GD']/20 * 0.5)
    s_ratio = h_quality / max(a_quality, 0.1)

    # SECTION 1: QUALITY ANALYSIS
    st.header("🔍 Match Balance Analysis")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(f"{home_team} Quality", f"{h_quality:.2f}",
                  help="Combined score of win rate and goal efficiency.")
    with col2:
        st.metric("Strength Gap", f"{s_ratio:.2f}x",
                  help="If this is above 1.0, the Home team is statistically superior.")
    with col3:
        st.metric(f"{away_team} Quality", f"{a_quality:.2f}",
                  help="Combined score of win rate and goal efficiency.")

    st.markdown("---")

    # SECTION 2: VISUALS & PREDICTION
    v_col1, v_col2 = st.columns([1.2, 1])

    with v_col1:
        st.subheader("📊 Performance Radar")

        categories = ['Attack Strength', 'Defense Solidity',
                      'League Rank', 'Venue Form', 'Overall Class']

        h_vals = [
            min(h_stats['Gls']/max(h_stats['Rank'], 1)*5, 10),
            max(10 - (h_stats['GA']/max(h_stats['Rank'], 1)*5), 0),
            max(21 - h_stats['Rank'], 1)/2,
            h_stats['H_Pts_MP']*3.33,
            h_stats['Pts_MP']*3.33
        ]

        a_vals = [
            min(a_stats['Gls']/max(a_stats['Rank'], 1)*5, 10),
            max(10 - (a_stats['GA']/max(a_stats['Rank'], 1)*5), 0),
            max(21 - a_stats['Rank'], 1)/2,
            a_stats['A_Pts_MP']*3.33,
            a_stats['Pts_MP']*3.33
        ]

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(r=h_vals, theta=categories,
                      fill='toself', name=home_team, line_color="#1c07df"))
        fig.add_trace(go.Scatterpolar(r=a_vals, theta=categories,
                      fill='toself', name=away_team, line_color="#ff2e0e"))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[
                                0, 10], gridcolor="gray", tickfont=dict(size=10)),
                angularaxis=dict(tickfont=dict(size=12, color="white"))
            ),
            showlegend=True,
            template="plotly_dark",
            margin=dict(l=50, r=50, t=20, b=20),
            height=450
        )
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("ℹ️ How to read this chart"):
            st.write("""
            **Scale: 0 (Poor) to 10 (Elite)**
            - **Attack Strength:** Based on goals scored relative to matches played.
            - **Defense Solidity:** Higher score means the team concedes fewer goals.
            - **League Rank:** Derived from current table position (1st = 10 pts).
            - **Venue Form:** Performance specific to Home ground (for Home team) or Away ground (for Away team).
            - **Overall Class:** Historical consistency across all matches.
            """)

    with v_col2:
        st.subheader("🔮 ML Prediction")
        st.info(
            "The model evaluates advanced underlying metrics including Shots on Target, corner pressure, clinical conversion rates, and cyclical momentum.")

        if st.button("CALCULATE PROBABILITIES", key="main_calc"):

            # Instantiate CustomData with ALL the newly engineered metrics
            data = CustomData(
                home_team=home_team, away_team=away_team,

                # Standard Metrics
                home_rank=h_stats['Rank'], home_pts_mp=h_stats['Pts_MP'],
                home_gd=h_stats['GD'], home_h_pts_mp=h_stats['H_Pts_MP'],
                home_gls=h_stats['Gls'], home_ga=h_stats['GA'], home_mp=h_stats['MP'],

                away_rank=a_stats['Rank'], away_pts_mp=a_stats['Pts_MP'],
                away_gd=a_stats['GD'], away_a_pts_mp=a_stats['A_Pts_MP'],
                away_gls=a_stats['Gls'], away_ga=a_stats['GA'], away_mp=a_stats['MP'],

                # --- NEW UNDERLYING METRICS ---
                home_sot_mp=h_stats['SoT_MP'], home_sot_conceded_mp=h_stats['SoT_Conceded_MP'],
                home_corners_mp=h_stats['Corners_MP'], home_cards_mp=h_stats['Cards_MP'],
                home_conversion=h_stats['Conversion'],

                away_sot_mp=a_stats['SoT_MP'], away_sot_conceded_mp=a_stats['SoT_Conceded_MP'],
                away_corners_mp=a_stats['Corners_MP'], away_cards_mp=a_stats['Cards_MP'],
                away_conversion=a_stats['Conversion'],

                # Form & Momentum
                home_form_pts=h_stats['Form_Pts'], home_form_gf=h_stats['Form_GF'], home_form_ga=h_stats['Form_GA'],
                home_form_sot=h_stats['Form_SoT'], home_form_sot_c=h_stats['Form_SoT_C'],

                away_form_pts=a_stats['Form_Pts'], away_form_gf=a_stats['Form_GF'], away_form_ga=a_stats['Form_GA'],
                away_form_sot=a_stats['Form_SoT'], away_form_sot_c=a_stats['Form_SoT_C']
            )

            df = data.get_data_as_dataframe()
            pipeline = PredictPipeline()
            result, probs = pipeline.predict(df)

            res_map = {'H': home_team, 'D': 'a Draw', 'A': away_team}

            st.markdown(f"### Result: **{res_map[result]}**")

            cols = st.columns(3)
            for i, (outcome, val) in enumerate(probs.items()):
                with cols[i]:
                    st.metric(outcome, f"{val:.1%}")
                    st.progress(val)


if __name__ == "__main__":
    main()
