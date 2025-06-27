# car_price_prediction/setup.py
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
    author="Your Name",
    author_email="your.email@example.com",
    description="Car price prediction using machine learning",
    long_description=open('README.md').read() if os.path.exists('README.md') else "",
    long_description_content_type="text/markdown",
    url="https://github.com/your-repo/car_price_prediction",
)