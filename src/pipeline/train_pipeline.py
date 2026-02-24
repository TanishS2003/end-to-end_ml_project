"""
Complete Training Pipeline
"""

import sys
from src.logger import logging
from src.exception import CustomException
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


def run_training_pipeline():
    try:
        logging.info("\n" + "="*70)
        logging.info("TRAINING PIPELINE STARTED")
        logging.info("="*70 + "\n")

        print("\n" + "="*70)
        print("⚽ LA LIGA MATCH PREDICTOR - TRAINING")
        print("="*70 + "\n")

        # Stage 1: Data Ingestion
        logging.info("STAGE 1: DATA INGESTION")
        print("📥 Stage 1: Downloading data from URLs...")

        ingestion = DataIngestion()
        raw_data_path = ingestion.initiate_data_ingestion()

        print(f"✅ Data downloaded: {raw_data_path}\n")

        # Stage 2: Data Transformation
        logging.info("STAGE 2: DATA TRANSFORMATION")
        print("🔄 Stage 2: Feature engineering (team strength + momentum)...")

        transformation = DataTransformation()
        train_arr, test_arr, preprocessor_path = transformation.initiate_data_transformation(
            raw_data_path)

        print(f"✅ Features engineered: {train_arr.shape[1]-1} features")
        print(f"✅ Train: {train_arr.shape[0]}, Test: {test_arr.shape[0]}\n")

        # Stage 3: Model Training
        logging.info("STAGE 3: MODEL TRAINING")
        print("🤖 Stage 3: Training 4 models (this takes 5-10 minutes)...")
        print("   Models: RandomForest, XGBoost, LightGBM, CatBoost\n")

        trainer = ModelTrainer()
        test_accuracy = trainer.initiate_model_trainer(train_arr, test_arr)

        print(f"\n✅ Best model test accuracy: {test_accuracy:.2%}")

        print("\n" + "="*70)
        print("🎉 TRAINING COMPLETED SUCCESSFULLY!")
        print("="*70)
        print(f"\n📁 Artifacts saved:")
        print(f"   - artifacts/model.pkl (best model)")
        print(f"   - artifacts/preprocessor.pkl")
        print(f"   - artifacts/model_comparison.csv")
        print("\n✅ Ready to run: streamlit run app.py")
        print("="*70 + "\n")

        return test_accuracy

    except Exception as e:
        logging.error(f"Pipeline failed: {str(e)}")
        print(f"\n❌ Error: {str(e)}\n")
        raise CustomException(e, sys)


if __name__ == "__main__":
    try:
        accuracy = run_training_pipeline()
    except Exception as e:
        print(f"❌ Fatal error: {str(e)}")
        sys.exit(1)
