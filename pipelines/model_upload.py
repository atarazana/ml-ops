from datetime import datetime
from os import environ

from boto3 import client

DEFAULT_DATA_OBJECT_NAME = 'live-data.csv'

DEFAULT_DATA_FOLDER = './data'
DEFAULT_DATA_FILE_NAME = 'data.csv'

DEFAULT_TRAIN_DATA_FILE_NAME = 'train-data.pkl'
DEFAULT_TRAIN_LABELS_FILE = 'train-labels.pkl'

DEFAULT_TEST_DATA_FILE_NAME = 'test-data.pkl'
DEFAULT_TEST_LABELS_FILE = 'test-labels.pkl'

DEFAULT_MODEL_FILE_NAME = 'model.joblib'

def upload_model(
        data_folder=DEFAULT_DATA_FOLDER,
        model_file_name=DEFAULT_MODEL_FILE_NAME):
    print('>>> Start model upload.')

    s3_endpoint_url = environ.get('AWS_S3_ENDPOINT')
    s3_access_key = environ.get('AWS_ACCESS_KEY_ID')
    s3_secret_key = environ.get('AWS_SECRET_ACCESS_KEY')
    s3_bucket_name = environ.get('AWS_S3_BUCKET')

    timestamp = datetime.now().strftime('%y%m%d%H%M')
    model_name = f'{timestamp}-{model_file_name}'

    print(f'  Uploading model to bucket {s3_bucket_name} '
          f'to S3 storage at {s3_endpoint_url}')

    s3_client = client(
        's3', endpoint_url=s3_endpoint_url,
        aws_access_key_id=s3_access_key, aws_secret_access_key=s3_secret_key
    )

    with open(f'{data_folder}/{model_file_name}', 'rb') as model_file:
        s3_client.upload_fileobj(model_file, s3_bucket_name, model_name)

    print('<<< Finished uploading model.')


if __name__ == '__main__':
    upload_model(
        data_folder='/data',
        model_file_name=DEFAULT_MODEL_FILE_NAME)
