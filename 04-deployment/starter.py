#!/usr/bin/env python
# coding: utf-8

import os
import sys
import pickle
import pandas as pd
import numpy as np


with open('model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)

categorical = ['PULocationID', 'DOLocationID']


def read_data(filename):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df

def prepare_dictionaries(df: pd.DataFrame):
    
    df[categorical] = df[categorical].astype(str)

    dicts = df[categorical].to_dict(orient='records')
    return dicts

def save_results(df, y_pred, output_file):
    df_result = pd.DataFrame()
    df_result['PULocationID'] = df['PULocationID']
    df_result['DOLocationID'] = df['DOLocationID']
    df_result['actual_duration'] = df['duration']
    df_result['predicted_duration'] = y_pred
    df_result['diff'] = df_result['actual_duration'] - df_result['predicted_duration']

    df_result.to_parquet(
            output_file,
            engine='pyarrow',
            compression=None,
            index=False
        )

def apply_model(input_file, output_file):
  
    df = read_data(input_file)
    dicts = prepare_dictionaries(df)

    X_val = dv.transform(dicts)
    y_pred = model.predict(X_val)

    save_results(df, y_pred, output_file)
    return output_file, y_pred


df = read_data('https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-03.parquet')

if __name__ == '__main__':

    year = int(sys.argv[1])
    month = int(sys.argv[2])

    input_file= f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet'

    base_dir = '/app'
    #output_dir = os.path.join(base_dir, 'yellow')
    output_file = os.path.join(base_dir, f'{year:04d}-{month:02d}.parquet')


    output_file, y_pred = apply_model(input_file, output_file)

    print(np.mean(y_pred))

