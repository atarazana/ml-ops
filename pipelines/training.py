

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression

from constants import DEFAULT_DATA_FOLDER, DEFAULT_TRAIN_DATA_FILE_NAME, DEFAULT_TRAIN_LABELS_FILE, DEFAULT_MODEL_FILE_NAME
from util.transformers import CombinedAttributesAdder


def train(
        data_folder=DEFAULT_DATA_FOLDER, 
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
            ("attribs_adder", CombinedAttributesAdder()),
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

    print('>>> Finished training model')


if __name__ == '__main__':
    train(data_folder=DEFAULT_DATA_FOLDER)
