# car_price_prediction/src/components/data_ingestion.py
import pandas as pd
import os
from src.logger import logging
from src.exception import CustomException
import sys

class DataIngestion:
    def __init__(self, data_path):
        self.data_path = data_path
        self.artifact_path = os.path.join('artifacts', 'cleaned_data.csv')

    def initiate_data_ingestion(self):
        logging.info("Starting data ingestion")
        try:
            # Load dataset
            cars = pd.read_csv(self.data_path)
            logging.info(f"Dataset loaded from {self.data_path}")

            # Clean data
            cars = cars[~cars['fuel_type'].isna()]
            cars['kms_driven'] = cars['kms_driven'].str.split().str.get(0).str.replace(',', '').astype(float)
            cars = cars[cars['Price'] != 'Ask For Price']
            cars['Price'] = cars['Price'].str.replace(',', '').astype(float)
            cars = cars[cars['year'].str.isnumeric()]
            cars['year'] = cars['year'].astype(int)
            cars['name'] = cars['name'].str.split().str.slice(0, 3).str.join(' ')
            cars = cars[cars['Price'] < 6000000].reset_index(drop=True)
            logging.info("Data cleaning completed")

            # Save cleaned data
            os.makedirs('artifacts', exist_ok=True)
            cars.to_csv(self.artifact_path, index=False)
            logging.info(f"Cleaned data saved to {self.artifact_path}")

            return cars
        except Exception as e:
            logging.error(f"Error in data ingestion: {str(e)}")
            raise CustomException(str(e), sys)

if __name__ == "__main__":
    data_ingestion = DataIngestion(data_path = r'/notebooks/quikr_car.csv')
    df = data_ingestion.initiate_data_ingestion()
    print(df.head())