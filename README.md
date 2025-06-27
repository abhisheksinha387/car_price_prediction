```markdown
# 🚗 Car Price Prediction

🎥 **[YouTube Video Tutorial](#)** | 🌐 **[Live Demo](#)**

## 📌 Overview

The **Car Price Prediction** project is a machine learning-based web application designed to estimate the price of used cars based on features like **car name, company, fuel type, year of manufacture**, and **kilometers driven**.

It uses a **Gradient Boosting Regressor** with Optuna hyperparameter tuning, integrated into a **Flask** web app for easy user interaction.

---

## 🗂 Project Structure

```

car\_price\_prediction/
├── notebooks/
│   ├── quikr\_car.csv
│   └── car\_price\_prediction.ipynb
├── src/
│   ├── components/
│   │   ├── data\_ingestion.py
│   │   ├── data\_transformation.py
│   │   └── model\_training.py
│   ├── pipeline/
│   │   ├── predict\_pipeline.py
│   │   └── train\_pipeline.py
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

````

---

## 🚀 Features

- **Data Ingestion**: Cleans the dataset (`quikr_car.csv`), handles missing values and outliers.
- **Feature Engineering**: Computes car age, kilometers per year, and premium brand status.
- **Model Training**: Trains a **Gradient Boosting Regressor** with **Optuna** tuning.
- **Prediction Pipeline**: Uses the trained model for real-time predictions.
- **Web Interface**: A user-friendly **Flask** app with dropdowns and input fields.
- **Logging & Exception Handling**: Custom logs and robust error tracking.

---

## 🧰 Requirements

Dependencies are listed in `requirements.txt`. Key libraries:

- `pandas`
- `numpy`
- `scikit-learn`
- `optuna`
- `xgboost`
- `lightgbm`
- `catboost`
- `flask`

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/abhisheksinha387/car_price_prediction.git
cd car_price_prediction
````

### 2. (Optional) Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. (Optional) Install as Package

```bash
python setup.py install
```

---

## 📈 Usage

### 🔧 Train the Model

Ensure `quikr_car.csv` is in the `notebooks/` folder.

```bash
python src/pipeline/train_pipeline.py
```

This will:

* Load and clean data
* Engineer features and preprocess
* Train and tune the model
* Save model and preprocessor to `artifacts/`

---

### 🌐 Run the Web App

```bash
python app.py
```

Then open [http://127.0.0.1:5000/](http://127.0.0.1:5000/) in your browser.

Input car details and get an estimated price instantly.

---

### 🤖 Predict Programmatically

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

## 📊 Dataset Details

The `quikr_car.csv` dataset includes:

* `name`: Car model
* `company`: Manufacturer
* `year`: Manufacturing year
* `Price`: Car price (target)
* `kms_driven`: Kilometers driven
* `fuel_type`: Petrol, Diesel, etc.


---

## 📦 Artifacts

Saved in the `artifacts/` directory:

* `cleaned_data.csv`: Processed dataset
* `preprocessor.pkl`: ColumnTransformer with OneHotEncoder + StandardScaler
* `model.pkl`: Trained Gradient Boosting model

---

## 📉 Model Performance

Evaluation metrics:

| Metric          | Value (approx) |
| --------------- | -------------- |
| R² Score (Test) | 0.85           |
| MAE             | ₹50,000        |
| RMSE            | ₹80,000        |
| CV R² Mean      | 0.83           |

---

## 🤝 Contributing

Contributions are welcome! 🚀

1. Fork the repo
2. Create a feature branch: `git checkout -b feature-name`
3. Commit your changes: `git commit -m 'Add feature'`
4. Push: `git push origin feature-name`
5. Open a pull request

---

## 📄 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for more details.

---

## 📬 Contact

* **Author**: Abhishek Sinha
* **Email**: [abhisheksinha.7742@gmail.com](mailto:abhisheksinha.7742@gmail.com)
* **GitHub**: [@abhisheksinha387](https://github.com/abhisheksinha387)

---


