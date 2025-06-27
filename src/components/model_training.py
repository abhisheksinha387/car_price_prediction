# car_price_prediction/src/components/model_training.py
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import cross_val_score
import optuna
import os
import pickle
from src.logger import logging
from src.exception import CustomException
import sys

class ModelTraining:
    def __init__(self):
        self.model_path = os.path.join('artifacts', 'model.pkl')

    def evaluate_model(self, true, predicted):
        try:
            mae = mean_absolute_error(true, predicted)
            rmse = np.sqrt(mean_squared_error(true, predicted))
            r2 = r2_score(true, predicted)
            return mae, rmse, r2
        except Exception as e:
            logging.error(f"Error in model evaluation: {str(e)}")
            raise CustomException(str(e), sys)

    def objective(self, trial, X_train, y_train):
        try:
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 10)
            }
            model = GradientBoostingRegressor(**params, random_state=42)
            score = cross_val_score(model, X_train, y_train, cv=5, scoring='r2').mean()
            return score
        except Exception as e:
            logging.error(f"Error in hyperparameter tuning: {str(e)}")
            raise CustomException(str(e), sys)

    def initiate_model_training(self, X_train, X_test, y_train, y_test):
        logging.info("Starting model training")
        try:
            # Hyperparameter tuning
            study = optuna.create_study(direction='maximize')
            study.optimize(lambda trial: self.objective(trial, X_train, y_train), n_trials=50)
            best_params = study.best_params
            logging.info(f"Best parameters: {best_params}")

            # Train model
            model = GradientBoostingRegressor(**best_params, random_state=42)
            model.fit(X_train, y_train)
            logging.info("Model trained")

            # Evaluate
            y_pred = model.predict(X_test)
            mae, rmse, r2 = self.evaluate_model(y_test, y_pred)
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
            logging.info(f"Test R2: {r2:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}, CV R2 Mean: {cv_scores.mean():.4f}")

            # Save model
            os.makedirs('artifacts', exist_ok=True)
            with open(self.model_path, 'wb') as file:
                pickle.dump(model, file)
            logging.info(f"Model saved to {self.model_path}")

            return {
                'Test R2': r2,
                'Test MAE': mae,
                'Test RMSE': rmse,
                'CV R2 Mean': cv_scores.mean(),
                'CV R2 Std': cv_scores.std() * 2,
                'Best Parameters': best_params
            }
        except Exception as e:
            logging.error(f"Error in model training: {str(e)}")
            raise CustomException(str(e), sys)

if __name__ == "__main__":
    from data_ingestion import DataIngestion
    from data_transformation import DataTransformation
    data_ingestion = DataIngestion(data_path=r'notebooks/quikr_car.csv')
    df = data_ingestion.initiate_data_ingestion()
    data_transformation = DataTransformation()
    X_train, X_test, y_train, y_test, _, _ = data_transformation.initiate_data_transformation(df)
    model_training = ModelTraining()
    results = model_training.initiate_model_training(X_train, X_test, y_train, y_test)
    print(results)