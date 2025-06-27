from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_training import ModelTraining
from src.logger import logging
from src.exception import CustomException
import sys

class TrainPipeline:
    def __init__(self, data_path):
        self.data_path = data_path

    def train(self):
        logging.info("Starting training pipeline")
        try:
            # Data ingestion
            data_ingestion = DataIngestion(data_path=self.data_path)
            df = data_ingestion.initiate_data_ingestion()

            # Data transformation
            data_transformation = DataTransformation()
            X_train, X_test, y_train, y_test, preprocessor, feature_columns = data_transformation.initiate_data_transformation(df)

            # Model training
            model_training = ModelTraining()
            results = model_training.initiate_model_training(X_train, X_test, y_train, y_test)

            logging.info("Training pipeline completed")
            return results
        except Exception as e:
            logging.error(f"Error in training pipeline: {str(e)}")
            raise CustomException(str(e), sys)

if __name__ == "__main__":
    train_pipeline = TrainPipeline(data_path=r'notebooks/quikr_car.csv')
    results = train_pipeline.train()
    print(results)