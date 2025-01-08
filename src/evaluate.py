import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle
import yaml
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix,f1_score, recall_score, classification_report
import os
import mlflow
from nltk.corpus import stopwords



os.environ['MLFLOW_TRACKING_URI']='https://dagshub.com/roshansalunke91/mlops1pipeline.mlflow'
os.environ['MLFLOW_TRACKING_USERNAME']='roshansalunke91'
os.environ['MLFLOW_TRACKING_PASSWORD']='19b31372c4920914da8021bca754c4f8e50ac529'


def evaluate():
    df = pd.read_csv(r'C:\Users\admin\Documents\Roshan\mlops1\data\raw\val.csv',sep=';',header=None)
    

    mlflow.set_tracking_uri('https://dagshub.com/roshansalunke91/mlops1pipeline.mlflow')

    model = pickle.load(open(r'C:\Users\admin\Documents\Roshan\mlops1\models\model.pkl','rb'))
    lemma = pickle.load(open(r'C:\Users\admin\Documents\Roshan\mlops1\models\lemma.pkl','rb'))
    tfidf = pickle.load(open(r'C:\Users\admin\Documents\Roshan\mlops1\models\tfidf.pkl','rb'))

    
    df = df.rename(columns={0:'text',1:'sentiment'})

    df['cleaned'] = df['text'].str.replace('[^a-zA-Z]', ' ', regex=True).str.lower()

    df['cleaned'] = df['cleaned'].str.split()

    df['lemma'] = df['cleaned'].apply(lambda words: ' '.join([lemma.lemmatize(word) for word in words if word not in set(stopwords.words('english'))]))

    df.drop('cleaned', axis=1, inplace=True)

    
    y = df['sentiment']
    x = tfidf.transform(df['lemma']).toarray()


    
    pred = model.predict(x)

    accuracy = accuracy_score(y, pred)
    mlflow.log_metric('acc',accuracy)
    print('model_acc ',accuracy)

if __name__=='__main__':
    evaluate()