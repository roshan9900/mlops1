import pandas as pd
import sys
import numpy as np
import yaml
import os

import nltk
import re
import string

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

## load parameter from params.yaml
params = yaml.safe_load(open('params.yaml'))['preprocess']

lemma = WordNetLemmatizer()
def preprocess(input_path, output_path):
    df = pd.read_csv(input_path,sep=';',header=None)
    df = df.rename(columns={0:'text',1:'sentiment'})

    df['cleaned'] = df['text'].str.replace('[^a-zA-Z]', ' ', regex=True).str.lower()

    df['cleaned'] = df['cleaned'].str.split()

    df['lemma'] = df['cleaned'].apply(lambda words: ' '.join([lemma.lemmatize(word) for word in words if word not in set(stopwords.words('english'))]))

    df.drop('cleaned', axis=1, inplace=True)

    tfidf = TfidfVectorizer(max_features=500, stop_words='english')
    x = tfidf.fit_transform(df['lemma']).toarray()

    xx = pd.DataFrame(x)
    xx['sentiment'] = df['sentiment']

    os.mkdir(os.path.dirname(output_path))
    xx.to_csv(output_path,index=False)
    print('success...........')


if __name__=='__main__':
    preprocess(r'C:\Users\admin\Documents\Roshan\mlops1\data\raw\train.csv',r'C:\Users\admin\Documents\Roshan\mlops1\data\preprocess\train.csv')