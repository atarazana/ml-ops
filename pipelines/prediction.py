import sys
import signal
import uuid
import os
from os import environ

import joblib
import sklearn
import pandas as pd

import onnxruntime
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression

# Python ≥3.5 is required
assert sys.version_info >= (3, 5)

# Scikit-Learn ≥0.20 is required
assert sklearn.__version__ >= "0.20"

DEFAULT_DATA_OBJECT_NAME = 'data-sets/live-data.csv'

DEFAULT_DATA_FOLDER = '/data'
DEFAULT_DATA_FILE_NAME = 'data.csv'

DEFAULT_TRAIN_DATA_FILE_NAME = 'train-data.pkl'
DEFAULT_TRAIN_LABELS_FILE = 'train-labels.pkl'

DEFAULT_TEST_DATA_FILE_NAME = 'test-data.pkl'
DEFAULT_TEST_LABELS_FILE = 'test-labels.pkl'

DEFAULT_MODEL_FILE_NAME = 'model.pkl'

DEFAULT_MODEL_OBJECT_PATH = 'models'
DEFAULT_MODEL_OBJECT_NAME = 'model.onnx'


def predict(
        data_folder=f'{environ.get("HOME")}/data',
        model_file_name=DEFAULT_MODEL_FILE_NAME):
    print('>>> Running prediction')

    
    MODEL_PATH=f'{data_folder}/{model_file_name}'
    print(f"MODEL_PATH={MODEL_PATH}")
    model = joblib.load(MODEL_PATH)

    payload = { 
         "longitude": -122.06,
         "latitude": 37.99,
         "housing_median_age": 17.0,
         "total_rooms": 1319.0,
         "total_bedrooms": 316.0,
         "population": 384.0,
         "households": 269.0,
         "median_income": 1.8229,
         "ocean_proximity": "NEAR BAY"
    }
    
    input_data = pd.DataFrame(payload, index=[0])
    print(input_data)
    prediction = model.predict(input_data).tolist()

    prediction_object = {
        "status": "success",
        "value": prediction,
        "valueType": "float",
        "explanation": "linear regressor value",
    }

    return prediction_object


def predict_onnx(
        data_folder=f'{environ.get("HOME")}/data',
        model_file_name=DEFAULT_MODEL_OBJECT_NAME):
    print('>>> Running prediction')

    
    model_path=f'{data_folder}/{model_file_name}'
    print(f"model_path={model_path}")

    payload = { 
         "longitude": -122.06,
         "latitude": 37.99,
         "housing_median_age": 17.0,
         "total_rooms": 1319.0,
         "total_bedrooms": 316.0,
         "population": 384.0,
         "households": 269.0,
         "median_income": 1.8229,
         "ocean_proximity": "NEAR BAY"
    }
    
    # Sample data for prediction
    data = [
        [-122.23, 37.88, 41.0, 880.0, 129.0, 322.0, 126.0, 8.3252],  # Numeric features
        ["NEAR BAY"]  # Categorical feature
    ]

    # Load the ONNX model
    sess = onnxruntime.InferenceSession(model_path)

    # Convert input data to a numpy array
    input_data = np.array(data)

    # Run inference
    output = sess.run(None, {"input": input_data})  # Assuming input name is "input"

    print(f"output={output}")
    
    prediction_object = {
        "status": "success",
        "value": "dummy",
        "valueType": "float",
        "explanation": "linear regressor value",
    }

    return prediction_object

if __name__ == '__main__':
    predict(data_folder=DEFAULT_DATA_FOLDER, model_file_name=DEFAULT_MODEL_FILE_NAME)
