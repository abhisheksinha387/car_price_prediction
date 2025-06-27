Here’s a properly formatted version of your project description in **GitHub README.md** Markdown syntax:

---

# 🚗 Car Price Prediction

**🎥 [YouTube Video Tutorial](#)** • **💻 [Live Demo](#)**

## 🧠 Overview

The Car Price Prediction project is a **machine learning-based web application** designed to predict the price of used cars based on features such as:

* Car name
* Company
* Fuel type
* Year of manufacture
* Kilometers driven

The model is built using a **Gradient Boosting Regressor**, includes data preprocessing and feature engineering, and is deployed via a **Flask web application** for user interaction.

---

## 📁 Project Structure

```
car_price_prediction/
├── notebooks/
│   ├── quikr_car.csv
│   └── car_price_prediction.ipynb
├── src/
│   ├── components/
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   └── model_training.py
│   ├── pipeline/
│   │   ├── predict_pipeline.py
│   │   └── train_pipeline.py
│   ├── exception.py
│   ├── logger.py
│   └── utils.py
├── templates/
│   ├── home.html
│   └── result.html
├── app.py
├── requirements.txt
├── setup.py
└── README.md
```

---

## 🚀 Features

* **Data Ingestion**: Cleans raw dataset, handles missing values, outliers, and converts data types.
* **Data Transformation**: Feature engineering and preprocessing (OneHotEncoder, StandardScaler).
* **Model Training**: Gradient Boosting Regressor with Optuna hyperparameter tuning.
* **Prediction Pipeline**: Flask interface for live predictions.
* **Web Interface**: Dropdowns and input fields for easy data entry.
* **Logging & Exception Handling**: Built-in for robustness.

---

## 📦 Requirements

Main libraries:

* `pandas`
* `numpy`
* `scikit-learn`
* `optuna`
* `xgboost`
* `lightgbm`
* `catboost`
* `flask`

Install using:

```bash
pip install -r requirements.txt
```

---

## 🛠️ Installation

1. **Clone the Repository**:

```bash
git clone https://github.com/abhisheksinha387/car_price_prediction.git
cd car_price_prediction
```

2. **Set Up Virtual Environment** *(optional)*:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install Dependencies**:

```bash
pip install -r requirements.txt
```

4. **Install Project (Optional)**:

```bash
python setup.py install
```

---

## 🏋️‍♂️ Model Training

Ensure `quikr_car.csv` is inside `notebooks/`.

Run:

```bash
python src/pipeline/train_pipeline.py
```

This will:

* Clean and transform the data
* Train the model with hyperparameter tuning
* Save `model.pkl` and `preprocessor.pkl` in `artifacts/`

---

## 🌐 Running the Web App

Start the Flask app:

```bash
python app.py
```

Then open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser.

Enter car details and get predicted price.

---

## 🧪 Predict Programmatically

```python
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

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
```

---

## 📊 Dataset

**File:** `notebooks/quikr_car.csv`

**Columns:**

* `name`: Car model name
* `company`: Manufacturer
* `year`: Year of manufacture
* `Price`: Target (in INR)
* `kms_driven`: Kilometers driven
* `fuel_type`: Fuel type

Preprocessing during ingestion handles missing values, outliers, and type issues.

---

## 📂 Artifacts

Generated in `artifacts/`:

* `cleaned_data.csv`: Cleaned data
* `preprocessor.pkl`: Preprocessing pipeline
* `model.pkl`: Trained ML model

---

## 📈 Model Performance

Metrics (approx):

| Metric     | Value     |
| ---------- | --------- |
| R² Score   | \~0.85    |
| MAE        | \~₹50,000 |
| RMSE       | \~₹80,000 |
| CV R² Mean | \~0.83    |

---

## 🤝 Contributing

1. Fork the repository
2. Create your branch: `git checkout -b feature-branch`
3. Commit your changes: `git commit -m 'Add feature'`
4. Push to branch: `git push origin feature-branch`
5. Open a pull request

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 📬 Contact

**Author**: Abhishek Sinha
📧 Email: [abhisheksinha.7742@gmail.com](mailto:abhisheksinha.7742@gmail.com)
🐙 GitHub: [@abhisheksinha387](https://github.com/abhisheksinha387)

---

