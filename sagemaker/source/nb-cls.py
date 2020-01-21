from sklearn.externals import joblib
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer

import os
import sys
import argparse
import subprocess
import pandas as pd
import numpy as np
from io import StringIO 
from six import BytesIO

from selector import Selector

from sagemaker_containers.beta.framework import worker, encoders


## Main start ##

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    class Helper:

        @staticmethod
        def install(package):
            subprocess.call([sys.executable, "-m", "pip", "install", package])

        @staticmethod
        def generate_train_test(df, target, train_size, random_state=42):
            """Generate train and test dataset. 

                :param pandas.DataFrame df: Dataset.
                :param string target: Name of target column.
                :param float train_size: Size of train proportion (0 <= train_size <= 1).
                :param int random_state: Seed for Random state. Default 42.
                :return: X_train, y_train, X_test, y_test.
                :rtype: tuple(pandas.DataFrame)
            """
            test_size = 1 - train_size
            columns = list(df.columns.values)
            columns.remove(target)

            train, teste = train_test_split(df, train_size=test_size, test_size=test_size, random_state=random_state)

            X_train = train[columns]
            y_train = train[[target]]

            X_test = teste[columns]
            y_test = teste[[target]]

            return (X_train, y_train, X_test, y_test)

        @staticmethod
        def create_tf_idf_pipeline(column):
            """Generate Tf-idf sklean pipeline in a column.

                :param string column: column to perform the transformation.
                :return: pipeline.
                :rtype: sklearn.pipeline.Pipeline
            """
            pipeline = Pipeline([
                (column, Selector(key=column)),
                ('tfidf', TfidfVectorizer(stop_words='english'))
            ])
            return pipeline

        @staticmethod
        def get_dataframe(columns):
            # Load the training data into a Pandas dataframe and make sure it is in the appropriate format
            df = pd.read_csv(args.train, sep='\t', error_bad_lines=False)
            df = df[columns]
            df = df.dropna()
            return df
        

    # Sagemaker specific arguments. Defaults are set in the environment variables.
    parser.add_argument('--output-data-dir', type=str, default='s3://sagemaker-product-classification-sklearn/nb/output_data')
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default='s3://sagemaker-product-classification-sklearn/data/raw/amazon_reviews_multilingual_UK_v1_00.tsv')

    args = parser.parse_args()
    
    columns = ['product_title', 'product_category', 'review_body', 'review_headline']
    target = 'product_category'
    train_size = 0.7
    
    Helper.install('s3fs')
    df = Helper.get_dataframe(columns)
    X_train, y_train, X_test, y_test = Helper.generate_train_test(df, target, train_size)
    del df
    
    
    product_title_pipeline = Helper.create_tf_idf_pipeline('product_title')
    review_body_pipeline = Helper.create_tf_idf_pipeline('review_body')
    review_headline_pipeline = Helper.create_tf_idf_pipeline('product_title')

    feats = FeatureUnion([
        ('product_title', product_title_pipeline), 
        ('review_body', review_body_pipeline),
        ('review_headline', review_headline_pipeline)
    ])

    nb_pipeline = Pipeline([
        ('features', feats),
        ('classifier', MultinomialNB(fit_prior=False)),
    ])
    
    model_nb = nb_pipeline.fit(X_train, y_train)
    print('model has been fitted')

    # Save the model to the output location in S3
    joblib.dump(model_nb, os.path.join(args.model_dir, "model.joblib"))


## Main end ##



def input_fn(request_body, request_content_type):
    """An input_fn that loads a pickled numpy array"""
    
    print(request_body)
    print(type(request_body))
    print(request_content_type)
    
    data = np.load(BytesIO(request_body))
    print(data)
    df = pd.DataFrame([data], columns=['product_title', 'review_body', 'review_headline'])
    print(df)
    return df
    
#     if request_content_type == 'application/x-npy':
#         data = np.load(request_body)
#         print(data)
#         df = pd.DataFrame(data, columns=['product_title', 'review_body', 'review_headline'])
#         print(df)
#         return df
#     else:
#         raise Exception('Invalid content_type. Try application/x-npy')

    
def output_fn(prediction_output, accept):
    print(prediction_output)
    print(type(prediction_output))
    if accept == 'application/x-npy':
        print('output_fn input is', prediction_output, 'in format', accept)
        buffer = BytesIO()
        np.save(buffer, prediction_output)
        return buffer.getvalue(), 'application/x-npy'
    elif accept == 'application/json':
        print('output_fn input is', prediction_output, 'in format', accept)
        return worker.Response(encoders.encode(prediction_output, accept), accept, mimetype=accept)
    else:
        raise ValueError('Accept header must be application/x-npy or application/json, but it is {}'.format(accept))


def predict_fn(input_data, model):
    prediction = model.predict(input_data)
    pred_prob = model.predict_proba(input_data)
    print(prediction, pred_prob)
    return np.array([prediction])


def model_fn(model_dir):
    clf = joblib.load(os.path.join(model_dir, "model.joblib"))
    return clf
