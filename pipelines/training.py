from os import environ

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression

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

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from boto3 import client
import onnx
from skl2onnx import to_onnx

rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6


# class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
#     def __init__(self, add_bedrooms_per_room=True):  # no *args or **kargs
#         self.add_bedrooms_per_room = add_bedrooms_per_room

#     def fit(self, X, y=None):
#         return self  # nothing else to do

#     def transform(self, X):
#         rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
#         population_per_household = X[:, population_ix] / X[:, households_ix]
#         if self.add_bedrooms_per_room:
#             bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
#             return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
#         else:
#             return np.c_[X, rooms_per_household, population_per_household]

#     # Implementing onnx_shape_calculator
#     def onnx_shape_calculator(self):
#         def shape_calculator(operator):
#             operator.outputs[0].type = operator.inputs[0].type
#         return shape_calculator
        
def train(
        data_folder=f'{environ.get("HOME")}/data',
        train_data_file_name=DEFAULT_TRAIN_DATA_FILE_NAME, 
        train_labels_file_name=DEFAULT_TRAIN_LABELS_FILE,
        model_file_name=DEFAULT_MODEL_FILE_NAME):
    print('>>> Training model')

    train_data = pd.read_pickle(f'{data_folder}/{train_data_file_name}')
    train_labels = pd.read_pickle(f'{data_folder}/{train_labels_file_name}')

    # Setting up num_pipeline with inputer, attr combiner (custom) and standard scaler
    num_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            # ("attribs_adder", CombinedAttributesAdder()),
            ("std_scaler", StandardScaler()),
        ]
    )

    num_attribs = list(train_data)
    num_attribs.remove("ocean_proximity")
    cat_attribs = ["ocean_proximity"]

    # Setting up full_pipeline with num_pipeline and OneHotEncoder for category attrs
    full_pipeline = ColumnTransformer(
        [
            ("num", num_pipeline, num_attribs),
            ("cat", OneHotEncoder(), cat_attribs),
        ]
    )

    # Setting up full_pipeline_with_predictor with full_pipeline and LinearRegression
    full_pipeline_with_predictor = Pipeline([("preparation", full_pipeline), ("linear", LinearRegression())])

    print("  Fitting pipeline")
    full_pipeline_with_predictor.fit(train_data, train_labels)

    print("  Dumping model to model.joblib")
    joblib.dump(full_pipeline_with_predictor, f'{data_folder}/{model_file_name}')

    # Transform the model file from pkl to onnx
    print("  Transform the model file from pkl to onnx")
    
    # Convert into ONNX format.
    onnx_model = to_onnx(full_pipeline_with_predictor, train_data[:1])

    s3_endpoint_url = environ.get('AWS_S3_ENDPOINT')
    s3_access_key = environ.get('AWS_ACCESS_KEY_ID')
    s3_secret_key = environ.get('AWS_SECRET_ACCESS_KEY')
    s3_bucket_name = environ.get('AWS_S3_BUCKET')

    s3_client = client(
        's3', endpoint_url=s3_endpoint_url,
        aws_access_key_id=s3_access_key, aws_secret_access_key=s3_secret_key
    )
    
    # Save the ONNX model to a file
    onnx.save_model(onnx_model, f'{data_folder}/{DEFAULT_MODEL_OBJECT_NAME}')

    with open(f'{data_folder}/{DEFAULT_MODEL_OBJECT_NAME}', 'rb') as model_file:
        s3_client.upload_fileobj(model_file, s3_bucket_name, f'{DEFAULT_MODEL_OBJECT_PATH}/{DEFAULT_MODEL_OBJECT_NAME}')

    print('>>> Finished training model')


if __name__ == '__main__':
    train(data_folder=DEFAULT_DATA_FOLDER)
