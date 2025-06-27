from flask import Flask, render_template, request
from src.pipeline.predict_pipeline import CustomData, PredictPipeline
from src.logger import logging
from src.exception import CustomException
import sys
import pandas as pd

app = Flask(__name__)

@app.route('/')
def home():
    try:
        # Load cleaned data for dropdowns
        df = pd.read_csv('artifacts/cleaned_data.csv')
        names = sorted(df['name'].unique().tolist())
        companies = sorted(df['company'].unique().tolist())
        fuel_types = sorted(df['fuel_type'].unique().tolist())
        logging.info("Loaded dropdown options")
        return render_template('home.html', names=names, companies=companies, fuel_types=fuel_types)
    except Exception as e:
        logging.error(f"Error in home route: {str(e)}")
        raise CustomException(str(e), sys)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        name = request.form['name']
        company = request.form['company']
        fuel_type = request.form['fuel_type']
        year = int(request.form['year'])
        kms_driven = float(request.form['kms_driven'])

        # Validate inputs
        if year < 1900 or year > 2025:
            raise ValueError("Year must be between 1900 and 2025")
        if kms_driven < 0:
            raise ValueError("Kilometers driven cannot be negative")

        # Create input data
        custom_data = CustomData(name, company, fuel_type, year, kms_driven)
        input_df = custom_data.get_data_as_dataframe()

        # Predict
        predict_pipeline = PredictPipeline()
        prediction = predict_pipeline.predict(input_df)

        logging.info(f"Prediction made: ₹{prediction:.2f}")
        return render_template('result.html', prediction=round(prediction, 2))
    except Exception as e:
        logging.error(f"Error in predict route: {str(e)}")
        return render_template('result.html', prediction=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)