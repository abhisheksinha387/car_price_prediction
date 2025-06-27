from setuptools import setup, find_packages
import os

setup(
    name="car_price_prediction",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        'pandas',
        'numpy',
        'scikit-learn',
        'optuna',
        'xgboost',
        'lightgbm',
        'catboost',
        'flask'
    ],
    author="Abhishek Sinha",
    author_email="abhisheksinha.7742@gmail.com",
    description="Car price prediction using machine learning",
    long_description=open('README.md').read() if os.path.exists('README.md') else "",
    long_description_content_type="text/markdown",
    url="https://github.com/abhisheksinha387/car_price_prediction",
)