"""
Complete Training Pipeline
This script runs the entire ML pipeline from data ingestion to model training
"""

import sys
from src.logger import logging
from src.exception import CustomException
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


def run_training_pipeline():
    """
    Execute the complete training pipeline with corrected data flow

    Flow:
    1. Data Ingestion: Read raw CSVs → Save raw data
    2. Data Transformation: Feature engineering on FULL data → Split → Preprocess
    3. Model Training: Train on processed data
    """
    try:
        logging.info("\n" + "="*70)
        logging.info("🚀 FOOTBALL MATCH PREDICTION - TRAINING PIPELINE")
        logging.info("="*70 + "\n")

        print("\n" + "="*70)
        print("🚀 FOOTBALL MATCH PREDICTION - TRAINING PIPELINE")
        print("="*70 + "\n")

        # ============================================================
        # STAGE 1: DATA INGESTION
        # ============================================================
        logging.info("="*70)
        logging.info(">>>>>> STAGE 1: DATA INGESTION STARTED <<<<<<")
        logging.info("="*70)

        print("📥 STAGE 1: Data Ingestion")
        print("-"*70)

        try:
            ingestion = DataIngestion()
            raw_data_path = ingestion.initiate_data_ingestion()

            logging.info("="*70)
            logging.info(">>>>>> STAGE 1: DATA INGESTION COMPLETED <<<<<<")
            logging.info(f">>>>>> Raw data saved at: {raw_data_path}")
            logging.info("="*70)
            logging.info("\n")

            print(f"✅ Stage 1 Complete: Raw data saved at {raw_data_path}\n")

        except Exception as e:
            logging.error("Stage 1 (Data Ingestion) failed")
            raise CustomException(e, sys)

        # ============================================================
        # STAGE 2: DATA TRANSFORMATION (includes splitting)
        # ============================================================
        logging.info("="*70)
        logging.info(">>>>>> STAGE 2: DATA TRANSFORMATION STARTED <<<<<<")
        logging.info("="*70)

        print("🔄 STAGE 2: Data Transformation")
        print("-"*70)

        try:
            transformation = DataTransformation()
            train_arr, test_arr, preprocessor_path = transformation.initiate_data_transformation(
                raw_data_path
            )

            logging.info("="*70)
            logging.info(
                ">>>>>> STAGE 2: DATA TRANSFORMATION COMPLETED <<<<<<")
            logging.info(f">>>>>> Train array shape: {train_arr.shape}")
            logging.info(f">>>>>> Test array shape: {test_arr.shape}")
            logging.info(f">>>>>> Preprocessor saved at: {preprocessor_path}")
            logging.info("="*70)
            logging.info("\n")

            print(f"✅ Stage 2 Complete:")
            print(f"   - Train samples: {train_arr.shape[0]}")
            print(f"   - Test samples: {test_arr.shape[0]}")
            print(f"   - Features: {train_arr.shape[1] - 1}\n")

        except Exception as e:
            logging.error("Stage 2 (Data Transformation) failed")
            raise CustomException(e, sys)

        # ============================================================
        # STAGE 3: MODEL TRAINING
        # ============================================================
        logging.info("="*70)
        logging.info(">>>>>> STAGE 3: MODEL TRAINING STARTED <<<<<<")
        logging.info("="*70)

        print("🤖 STAGE 3: Model Training")
        print("-"*70)

        try:
            trainer = ModelTrainer()
            test_accuracy = trainer.initiate_model_trainer(train_arr, test_arr)

            logging.info("="*70)
            logging.info(">>>>>> STAGE 3: MODEL TRAINING COMPLETED <<<<<<")
            logging.info(f">>>>>> Test Accuracy: {test_accuracy:.4f}")
            logging.info("="*70)
            logging.info("\n")

            print(f"✅ Stage 3 Complete: Test Accuracy = {test_accuracy:.2%}\n")

        except Exception as e:
            logging.error("Stage 3 (Model Training) failed")
            raise CustomException(e, sys)

        # ============================================================
        # PIPELINE SUMMARY
        # ============================================================
        logging.info("="*70)
        logging.info("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        logging.info("="*70)
        logging.info("Summary:")
        logging.info(f"  - Data ingestion: ✓")
        logging.info(f"  - Data transformation: ✓")
        logging.info(f"  - Model training: ✓")
        logging.info(f"  - Final test accuracy: {test_accuracy:.4f}")
        logging.info("="*70)

        print("\n" + "="*70)
        print("🎉 PIPELINE COMPLETED SUCCESSFULLY! 🎉")
        print("="*70)
        print(f"\n📊 Final Model Performance:")
        print(f"   Test Accuracy: {test_accuracy:.2%}")
        print(f"\n📁 Artifacts saved in 'artifacts/' directory:")
        print(f"   - data.csv (raw concatenated data)")
        print(f"   - train.csv (training split - AFTER feature engineering)")
        print(f"   - test.csv (test split - AFTER feature engineering)")
        print(f"   - preprocessor.pkl (preprocessing pipeline)")
        print(f"   - model.pkl (trained model)")
        print(f"\n✅ Key Features of This Pipeline:")
        print(f"   - Feature engineering happens BEFORE split (no data leakage)")
        print(f"   - Train and test share same feature engineering context")
        print(f"   - Team statistics are cumulative across full dataset")
        print(f"   - Bayesian optimization for hyperparameter tuning")
        print(f"   - Production-ready with proper logging and error handling")
        print("="*70 + "\n")

        return test_accuracy

    except Exception as e:
        logging.error("="*70)
        logging.error("❌ PIPELINE FAILED")
        logging.error("="*70)
        logging.error(f"Error: {str(e)}")
        logging.error("="*70)

        print("\n" + "="*70)
        print("❌ PIPELINE FAILED")
        print("="*70)
        print(f"Error: {str(e)}")
        print("\nCheck logs for detailed error information.")
        print("="*70 + "\n")

        raise CustomException(e, sys)


if __name__ == "__main__":
    try:
        logging.info("="*70)
        logging.info("Starting Training Pipeline Execution")
        logging.info("="*70)

        print("\n🚀 Starting Football Match Prediction Training Pipeline...\n")

        accuracy = run_training_pipeline()

        logging.info("="*70)
        logging.info(f"Pipeline execution completed successfully")
        logging.info(f"Final test accuracy: {accuracy:.4f}")
        logging.info("="*70)

        print(
            f"\n✅ Pipeline execution completed with test accuracy: {accuracy:.2%}\n")

    except Exception as e:
        logging.error("="*70)
        logging.error("Pipeline execution failed at top level")
        logging.error(f"Error: {str(e)}")
        logging.error("="*70)

        print(f"\n❌ Fatal error: {str(e)}\n")
        sys.exit(1)
