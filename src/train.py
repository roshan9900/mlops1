import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle
import yaml
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix, recall_score, classification_report
import mlflow
from mlflow.models import infer_signature
import os
from sklearn.model_selection import train_test_split
from urllib.parse import urlparse

os.environ['MLFLOW_TRACKING_URI']='https://dagshub.com/roshansalunke91/mlops1pipeline.mlflow'
os.environ['MLFLOW_TRACKING_USERNAME']='roshansalunke91'
os.environ['MLFLOW_TRACKING_PASSWORD']='19b31372c4920914da8021bca754c4f8e50ac529'

def building_model(x_train, y_train):
    rf = RandomForestClassifier(random_state=42)
    rf.fit(x_train, y_train)
    return rf

def train(data_path, model_path, n_estimators, max_depth):
    data = pd.read_csv(data_path)
    x = data.drop('sentiment',axis=1)
    y = data['sentiment']

    mlflow.set_tracking_uri('https://dagshub.com/roshansalunke91/mlops1pipeline.mlflow')

    with mlflow.start_run():
        x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=.2,random_state=42)
        signature = infer_signature(x_train, y_train)

        rf = building_model(x_train, y_train)

        pred = rf.predict(x_test)

        accuracy = accuracy_score(y_test, pred)
        precision = precision_score(y_test ,pred)
        recall = recall_score(y_test, pred)
        print(accuracy, precision, recall)
