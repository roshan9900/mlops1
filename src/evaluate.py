import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle
import yaml
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix,f1_score, recall_score, classification_report
import os
import mlflow


os.environ['MLFLOW_TRACKING_URI']='https://dagshub.com/roshansalunke91/mlops1pipeline.mlflow'
os.environ['MLFLOW_TRACKING_USERNAME']='roshansalunke91'
os.environ['MLFLOW_TRACKING_PASSWORD']='19b31372c4920914da8021bca754c4f8e50ac529'


def evaluate():
    data = pd.read_csv(r'C:\Users\admin\Documents\Roshan\mlops1\data\raw\val.csv')
    x = data.drop('sentiment',axis=1)
    y = data['sentiment']

    mlflow.set_tracking_uri('https://dagshub.com/roshansalunke91/mlops1pipeline.mlflow')

    model = pickle.load(open(r'C:\Users\admin\Documents\Roshan\mlops1\models\model.pkl','rb'))

    pred = model.predict(x)

    accuracy = accuracy_score(y, pred)
    mlflow.log_metric('acc',accuracy)
    print('model_acc ',accuracy)

if __name__=='__main__':
    evaluate()