import numpy as np

from numpy import save
from pandas import read_csv

import pandas as pd

# from sklearn.datasets import load_iris
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression

from constants import DEFAULT_DATA_FILE_NAME, DEFAULT_DATA_FOLDER, DEFAULT_TEST_DATA_FILE_NAME, DEFAULT_TRAIN_DATA_FILE_NAME, DEFAULT_TRAIN_LABELS_FILE
from constants import DEFAULT_TEST_LABELS_FILE


def preprocess(data_folder=DEFAULT_DATA_FOLDER,data_file=DEFAULT_DATA_FILE_NAME):
    print('>>> Preprocessing data')

    data = read_csv(f'{data_folder}/{data_file}')

    # Create bins for stratified shuffling of data and fill the income cat with those numeric categories
    data["income_cat"] = pd.cut(data["median_income"], bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf], labels=[1, 2, 3, 4, 5])

    # Shuffle and splitting into train and test
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(data, data["income_cat"]):
        strat_train_set = data.loc[train_index]
        strat_test_set = data.loc[test_index]

    # This is the 'y' or label
    train_data_labels = strat_train_set["median_house_value"].copy()
    test_data_labels = strat_test_set["median_house_value"].copy()

    # Save labels to file
    train_data_labels.to_pickle(f'{data_folder}/{DEFAULT_TRAIN_LABELS_FILE}')
    test_data_labels.to_pickle(f'{data_folder}/{DEFAULT_TEST_LABELS_FILE}')

    # print("  Dropping category columns => (ocean_proximity)")
    # strat_train_set = strat_train_set.drop("ocean_proximity", axis=1)
    # strat_test_set = strat_test_set.drop("ocean_proximity", axis=1)

    strat_train_set = strat_train_set.drop(
        "median_house_value", axis=1
    ).drop(
        "income_cat", axis=1
    )
    # .drop(
    #     "ocean_proximity", axis=1)
    
    strat_test_set = strat_test_set.drop(
        "median_house_value", axis=1
    ).drop(
        "income_cat", axis=1
    )
    # .drop(
    #     "ocean_proximity", axis=1)

    print("  Saving training data")
    strat_train_set.to_pickle(f'{data_folder}/{DEFAULT_TRAIN_DATA_FILE_NAME}')
    print("  Saving test data")
    strat_test_set.to_pickle(f'{data_folder}/{DEFAULT_TEST_DATA_FILE_NAME}')

    print('>>> Finished processing')


if __name__ == '__main__':
    preprocess(data_folder='/data')
