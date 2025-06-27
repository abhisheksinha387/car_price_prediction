import pickle
import os
from src.logger import logging
from src.exception import CustomException
import sys

def load_object(file_path):
    try:
        with open(file_path, 'rb') as file:
            obj = pickle.load(file)
        logging.info(f"Loaded object from {file_path}")
        return obj
    except Exception as e:
        logging.error(f"Error loading object from {file_path}: {str(e)}")
        raise CustomException(str(e), sys)

def save_object(file_path, obj):
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as file:
            pickle.dump(obj, file)
        logging.info(f"Saved object to {file_path}")
    except Exception as e:
        logging.error(f"Error saving object to {file_path}: {str(e)}")
        raise CustomException(str(e), sys)