from os import environ

import boto3

DEFAULT_DATA_OBJECT_NAME = 'live-data.csv'

DEFAULT_DATA_FOLDER = './data'
DEFAULT_DATA_FILE_NAME = 'data.csv'

DEFAULT_TRAIN_DATA_FILE_NAME = 'train-data.pkl'
DEFAULT_TRAIN_LABELS_FILE = 'train-labels.pkl'

DEFAULT_TEST_DATA_FILE_NAME = 'test-data.pkl'
DEFAULT_TEST_LABELS_FILE = 'test-labels.pkl'

DEFAULT_MODEL_FILE_NAME = 'model.joblib'

# Downloads data data_object_name from AWS S3 bucket and leaves it in data_folder=DEFAULT_DATA_FOLDER
def ingest_data(data_object_name='', data_folder=DEFAULT_DATA_FOLDER, data_file=DEFAULT_DATA_FILE_NAME):
    print('>>> Starting data ingestion.')

    s3_endpoint_url = environ.get('AWS_S3_ENDPOINT')
    s3_access_key = environ.get('AWS_ACCESS_KEY_ID')
    s3_secret_key = environ.get('AWS_SECRET_ACCESS_KEY')
    s3_bucket_name = environ.get('AWS_S3_BUCKET')

    data_object_name = data_object_name or environ.get(
        'DATA_OBJECT_NAME', DEFAULT_DATA_OBJECT_NAME
    )

    print(f'  ingest_data(data_object_name={data_object_name} and data_folder={data_folder})')

    print(f'  Downloading data "{data_object_name}" '
          f'from bucket "{s3_bucket_name}" '
          f'from S3 storage at {s3_endpoint_url} '
          f'to {data_folder}/{data_file}')

    s3_client = boto3.client(
        's3', endpoint_url=s3_endpoint_url,
        aws_access_key_id=s3_access_key, aws_secret_access_key=s3_secret_key
    )

    s3_client.download_file(
        s3_bucket_name,
        data_object_name,
        f'{data_folder}/{data_file}'
    )
    print('<<< Finished data ingestion.')


if __name__ == '__main__':
    ingest_data(data_folder='/data')
