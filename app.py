from src.logger import logging
from src.exception import CustomException
from src.pipeline.predict_pipeline import PredictPipeline, CustomData
import streamlit as st
import pandas as pd
import sys
import os

# Add parent directory to path to import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# Page configuration
st.set_page_config(
    page_title="La Liga Match Predictor",
    page_icon="⚽",
    layout="centered"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #FF4B4B;
        color: white;
        font-weight: bold;
        padding: 0.75rem;
        font-size: 1.1rem;
    }
    .result-text {
        font-size: 2rem;
        font-weight: bold;
        text-align: center;
        margin: 1rem 0;
    }
    .prob-text {
        font-size: 1.2rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Spanish La Liga teams
TEAMS = [
    'Alaves', 'Almeria', 'Ath Bilbao', 'Ath Madrid', 'Barcelona',
    'Betis', 'Cadiz', 'Celta', 'Elche', 'Espanol', 'Getafe',
    'Girona', 'Granada', 'Las Palmas', 'Leganes', 'Levante',
    'Mallorca', 'Osasuna', 'Oviedo', 'Real Madrid', 'Sevilla',
    'Sociedad', 'Valencia', 'Valladolid', 'Vallecano', 'Villarreal'
]

# Header
st.title("⚽ La Liga Match Predictor")
st.markdown(
    "### Predict the outcome of Spanish La Liga matches using Machine Learning")
st.markdown("---")

# Sidebar with information
with st.sidebar:
    st.header("ℹ️ About")
    st.info(
        """
        This app predicts the outcome of Spanish La Liga football matches 
        using a Random Forest model trained on historical match data.
        
        **How it works:**
        1. Select the home team
        2. Select the away team
        3. Click 'Predict Match Outcome'
        4. View the prediction and probabilities
        
        **Predictions:**
        - **H** = Home Win
        - **D** = Draw
        - **A** = Away Win
        """
    )

    st.header("📊 Model Info")
    st.markdown("""
        - **Algorithm:** Random Forest Classifier
        - **Optimization:** Bayesian Search
        - **Features:** Team rankings, form, goals, etc.
        - **Training Data:** 4 seasons of La Liga matches
    """)

    st.header("⚠️ Disclaimer")
    st.warning(
        """
        This is a prediction model for educational purposes. 
        Football matches are unpredictable and many factors 
        can influence the outcome. Use predictions responsibly.
        """
    )

# Main content
col1, col2 = st.columns(2)

with col1:
    st.subheader("🏠 Home Team")
    home_team = st.selectbox(
        "Select Home Team",
        options=sorted(TEAMS),
        index=sorted(TEAMS).index('Barcelona'),
        key='home'
    )

with col2:
    st.subheader("✈️ Away Team")
    away_team = st.selectbox(
        "Select Away Team",
        options=sorted(TEAMS),
        index=sorted(TEAMS).index('Real Madrid'),
        key='away'
    )

# Validation
if home_team == away_team:
    st.error("⚠️ Please select different teams for home and away!")
    st.stop()

# Display selected match
st.markdown("---")
st.markdown(f"### 🎯 Selected Match")
st.markdown(f"## **{home_team}** 🆚 **{away_team}**")
st.markdown("---")

# Predict button
if st.button("🔮 Predict Match Outcome", type="primary"):
    try:
        logging.info("="*70)
        logging.info(f"Prediction requested: {home_team} vs {away_team}")
        logging.info("="*70)

        with st.spinner('🤖 Analyzing match data and making prediction...'):
            # Create custom data
            custom_data = CustomData(
                home_team=home_team,
                away_team=away_team
            )

            logging.info("Custom data object created")

            # Convert to DataFrame
            pred_df = custom_data.get_data_as_dataframe()
            logging.info(f"Data converted to DataFrame: {pred_df.shape}")

            # Make prediction
            predict_pipeline = PredictPipeline()
            prediction, probabilities = predict_pipeline.predict(pred_df)

            logging.info(f"Prediction completed: {prediction}")
            logging.info(f"Probabilities: {probabilities}")

        # Display results
        st.success("✅ Prediction completed!")

        logging.info("Displaying results to user")

        # Display prediction with larger text
        result_map = {
            'H': f'🏆 {home_team} Win',
            'D': '🤝 Draw',
            'A': f'🏆 {away_team} Win'
        }

        result_emoji = {
            'H': '🟢',
            'D': '🟡',
            'A': '🔵'
        }

        # Main prediction display
        st.markdown(f'<p class="result-text">{result_emoji[prediction]} Predicted: {result_map[prediction]}</p>',
                    unsafe_allow_html=True)

        st.markdown("---")

        # Display probabilities
        st.markdown("### 📊 Prediction Probabilities")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                label=f"🏠 {home_team} Win",
                value=f"{probabilities['Home Win']:.1%}",
                delta=None
            )

        with col2:
            st.metric(
                label="🤝 Draw",
                value=f"{probabilities['Draw']:.1%}",
                delta=None
            )

        with col3:
            st.metric(
                label=f"✈️ {away_team} Win",
                value=f"{probabilities['Away Win']:.1%}",
                delta=None
            )

        # Probability bar chart
        st.markdown("---")
        prob_df = pd.DataFrame({
            'Outcome': [f'{home_team} Win', 'Draw', f'{away_team} Win'],
            'Probability': [
                probabilities['Home Win'],
                probabilities['Draw'],
                probabilities['Away Win']
            ]
        })

        st.bar_chart(prob_df.set_index('Outcome'))

        # Confidence indicator
        max_prob = max(probabilities.values())
        if max_prob > 0.6:
            st.success("🎯 High confidence prediction!")
            logging.info("High confidence prediction displayed")
        elif max_prob > 0.45:
            st.info("📊 Moderate confidence prediction")
            logging.info("Moderate confidence prediction displayed")
        else:
            st.warning("⚠️ Low confidence - match could go either way!")
            logging.info("Low confidence prediction displayed")

        logging.info("="*70)
        logging.info("Prediction display completed successfully")
        logging.info("="*70)

    except FileNotFoundError as e:
        error_msg = f"Model files not found: {str(e)}"
        logging.error(error_msg)
        st.error("""
            ❌ Model files not found! 
            
            Please make sure you have trained the model first by running:
            ```
            python train_pipeline.py
            ```
            
            This will create the required model files in the 'artifacts/' directory.
        """)
    except CustomException as e:
        error_msg = f"CustomException occurred: {str(e)}"
        logging.error(error_msg)
        st.error(f"❌ An error occurred: {str(e)}")
        st.error("Please check the logs for more details.")
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        logging.error(error_msg)
        logging.error(f"Error type: {type(e).__name__}")
        st.error(f"❌ An unexpected error occurred: {str(e)}")
        st.error("Please check the logs for more details.")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666; padding: 2rem;'>
        <p>⚽ Built with Streamlit and Machine Learning</p>
        <p>Data: Spanish La Liga (2022-2026)</p>
    </div>
""", unsafe_allow_html=True)
