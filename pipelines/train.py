
import sys

import sklearn

# Common imports
import numpy as np
import os

import pandas as pd

# from sklearn.datasets import load_iris
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression

import joblib

from util import CombinedAttributesAdder

# Format the JSON object as a string
import json

# Python ≥3.5 is required
assert sys.version_info >= (3, 5)
# Scikit-Learn ≥0.20 is required
assert sklearn.__version__ >= "0.20"


if "MODEL_PATH" in os.environ:
    pass
else:
    print("MODEL_PATH CANNOT BE EMPTY!!!")
    sys.exit()

MODEL_PATH = os.getenv("MODEL_PATH")

print(f"MODEL_PATH={MODEL_PATH}")

HOUSING_PATH = os.path.join("datasets", "housing")


def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


print("Loading data")
housing = load_housing_data()

housing["income_cat"] = pd.cut(housing["median_income"], bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf], labels=[1, 2, 3, 4, 5])

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

housing = strat_train_set.drop("median_house_value", axis=1).drop(
    "income_cat", axis=1
)  # drop 'y' for training set (predictors/features)
housing_labels = strat_train_set["median_house_value"].copy()  # 'y'

print("Dropping category (string) columns")
housing_num = housing.drop("ocean_proximity", axis=1)

print("Setting up num_pipeline with inputer, attr combiner (custom) and standard scaler")
num_pipeline = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="median")),
        ("attribs_adder", CombinedAttributesAdder()),
        ("std_scaler", StandardScaler()),
    ]
)

print(f"all_attribs={list(housing)}")
num_attribs = list(housing_num)
print(f"num_attribs={num_attribs}")
cat_attribs = ["ocean_proximity"]
print(f"num_attribs={num_attribs}")


print("Setting up full_pipeline with num_pipeline and OneHotEncoder for category attrs")
full_pipeline = ColumnTransformer(
    [
        ("num", num_pipeline, num_attribs),
        ("cat", OneHotEncoder(), cat_attribs),
    ]
)

# full_pipeline.named_transformers_["cat"].handle_unknown = 'ignore'

print("Setting up full_pipeline_with_predictor with full_pipeline and LinearRegression")
full_pipeline_with_predictor = Pipeline([("preparation", full_pipeline), ("linear", LinearRegression())])

print("Fitting pipeline")
print(f"all_attribs_2={list(housing)}")
full_pipeline_with_predictor.fit(housing, housing_labels)

print("Dumping model to model.joblib")
joblib.dump(full_pipeline_with_predictor, MODEL_PATH)

print("Test data")
sample = strat_test_set.sample()
test_sample = sample.drop("median_house_value", axis=1).drop("income_cat", axis=1)
# test_set = strat_test_set.drop("median_house_value", axis=1).drop("income_cat", axis=1)
# sample = test_set.sample()
# type(strat_test_set.iloc[sample.index[0]])
# print(sample)
# print(test_sample)

# Convert the random row to a JSON object
json_object = test_sample.to_dict(orient="records")[0]

json_string = json.dumps(json_object, indent=4)

# Print the cURL command
print(f'Run the next command it should predict a value close to: {sample["median_house_value"].values[0]}')
print(f"curl -X POST -H \"Content-Type: application/json\" -d '{json_string}' http://localhost:8080/predict")
