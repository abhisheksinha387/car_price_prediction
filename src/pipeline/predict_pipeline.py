import pandas as pd
import pickle
import os
from src.logger import logging
from src.exception import CustomException
import sys

class PredictPipeline:
    def __init__(self):
        self.model_path = os.path.join('artifacts', 'model.pkl')
        self.preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')

    def load_artifacts(self):
        try:
            with open(self.preprocessor_path, 'rb') as file:
                preprocessor = pickle.load(file)
            with open(self.model_path, 'rb') as file:
                model = pickle.load(file)
            logging.info("Artifacts loaded")
            return preprocessor, model
        except Exception as e:
            logging.error(f"Error loading artifacts: {str(e)}")
            raise CustomException(str(e), sys)

    def predict(self, input_data):
        logging.info("Starting prediction")
        try:
            preprocessor, model = self.load_artifacts()
            input_transformed = preprocessor.transform(input_data)
            prediction = model.predict(input_transformed)[0]
            logging.info(f"Prediction: {prediction}")
            return prediction
        except Exception as e:
            logging.error(f"Error in prediction: {str(e)}")
            raise CustomException(str(e), sys)

class CustomData:
    def __init__(self, name, company, fuel_type, year, kms_driven):
        self.name = name
        self.company = company
        self.fuel_type = fuel_type
        self.year = int(year)
        self.kms_driven = float(kms_driven)

    def get_data_as_dataframe(self):
        try:
            car_age = 2025 - self.year
            kms_per_year = self.kms_driven / (car_age + 1)
            is_premium_brand = 1 if self.company in ['Audi', 'BMW', 'Mercedes', 'Jaguar', 'Volvo'] else 0
            data = pd.DataFrame({
                'name': [self.name],
                'company': [self.company],
                'fuel_type': [self.fuel_type],
                'car_age': [car_age],
                'kms_per_year': [kms_per_year],
                'is_premium_brand': [is_premium_brand]
            })
            logging.info("Input data converted to DataFrame")
            return data
        except Exception as e:
            logging.error(f"Error in data conversion: {str(e)}")
            raise CustomException(str(e), sys)

if __name__ == "__main__":
    custom_data = CustomData(
        name="Maruti Suzuki Swift",
        company="Maruti",
        fuel_type="Petrol",
        year=2019,
        kms_driven=100
    )
    input_df = custom_data.get_data_as_dataframe()
    predict_pipeline = PredictPipeline()
    prediction = predict_pipeline.predict(input_df)
    print(f"Predicted Price: ₹{prediction:.2f}")