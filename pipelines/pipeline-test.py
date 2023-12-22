from os import environ

from data_ingestion import ingest_data
from model_upload import upload_model
from training import train
from preprocessing import preprocess

# # Name of the object in the bucket where the data is. Default it live-data.csv
# data_object_name = environ.get('DATA_OBJECT_NAME', 'live-data.csv')
# data_folder = environ.get('DATA_FOLDER', './data')

# Ingest data from bucket
print('1. INGEST DATA')
print('#####################################')
ingest_data()

print('2. PREPROCESS')
print('#####################################')
preprocess()

print('3. TRAIN')
print('#####################################')
train()

print('4. UPLOAD MODEL')
print('#####################################')
upload_model()
