# car_price_prediction/src/components/data_transformation.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import os
import pickle
from src.logger import logging
from src.exception import CustomException
import sys

class DataTransformation:
    def __init__(self):
        self.preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
        self.categorical_features = ['name', 'company', 'fuel_type']
        self.numerical_features = ['car_age', 'kms_per_year', 'is_premium_brand']

    def get_data_transformer(self):
        try:
            logging.info("Creating preprocessor")
            preprocessor = ColumnTransformer(
                transformers=[
                    ('cat', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False), self.categorical_features),
                    ('num', StandardScaler(), self.numerical_features)
                ],
                remainder='passthrough'
            )
            return preprocessor
        except Exception as e:
            logging.error(f"Error in creating preprocessor: {str(e)}")
            raise CustomException(str(e), sys)

    def initiate_data_transformation(self, df):
        logging.info("Starting data transformation")
        try:
            # Feature engineering
            df['car_age'] = 2025 - df['year']
            df['kms_per_year'] = df['kms_driven'] / (df['car_age'] + 1)
            df['is_premium_brand'] = df['company'].isin(['Audi', 'BMW', 'Mercedes', 'Jaguar', 'Volvo']).astype(int)
            logging.info("Feature engineering completed")

            # Define features and target
            X = df[['name', 'company', 'fuel_type', 'car_age', 'kms_per_year', 'is_premium_brand']]
            y = df['Price']

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            logging.info("Data split into train and test sets")

            # Apply preprocessing
            preprocessor = self.get_data_transformer()
            X_train_transformed = preprocessor.fit_transform(X_train)
            X_test_transformed = preprocessor.transform(X_test)
            logging.info("Preprocessing applied")

            # Save preprocessor
            os.makedirs('artifacts', exist_ok=True)
            with open(self.preprocessor_path, 'wb') as file:
                pickle.dump(preprocessor, file)
            logging.info(f"Preprocessor saved to {self.preprocessor_path}")

            return X_train_transformed, X_test_transformed, y_train, y_test, preprocessor, X_train.columns
        except Exception as e:
            logging.error(f"Error in data transformation: {str(e)}")
            raise CustomException(str(e), sys)

if __name__ == "__main__":
    from data_ingestion import DataIngestion
    data_ingestion = DataIngestion(data_path= r'notebooks/quikr_car.csv')
    df = data_ingestion.initiate_data_ingestion()
    data_transformation = DataTransformation()
    X_train, X_test, y_train, y_test, preprocessor, feature_columns = data_transformation.initiate_data_transformation(df)
    print(X_train.shape, X_test.shape)