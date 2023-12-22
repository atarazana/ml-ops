#!/bin/sh

export MODEL_PATH=./models/model.joblib

export AWS_S3_ENDPOINT=https://minio-api-minio.apps.daedalus.sandbox242.opentlc.com
export AWS_ACCESS_KEY_ID=minio
export AWS_SECRET_ACCESS_KEY=minio123
export AWS_S3_BUCKET=ml-ops

python pipeline-test.py