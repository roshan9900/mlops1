import streamlit as st
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


model_path = r'C:\Users\admin\Documents\Roshan\mlops1\models\model.pkl'
lemma_path = r'C:\Users\admin\Documents\Roshan\mlops1\models\lemma.pkl'
tfidf_path = r'C:\Users\admin\Documents\Roshan\mlops1\models\tfidf.pkl'

model = pickle.load(open(model_path, 'rb'))
lemma = pickle.load(open(lemma_path, 'rb'))
tfidf = pickle.load(open(tfidf_path, 'rb'))

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Preprocessing function
def preprocess_text(text):
    # Clean text
    cleaned_text = text.lower().replace('[^a-zA-Z]', ' ')
    words = cleaned_text.split()
    # Remove stopwords and lemmatize
    processed_text = ' '.join(
        [lemmatizer.lemmatize(word) for word in words if word not in set(stopwords.words('english'))]
    )
    return processed_text

# Streamlit App
st.title("Sentiment Prediction App")

# Input text
input_text = st.text_area("Enter text to analyze sentiment:")

if st.button("Predict Sentiment"):
    if input_text.strip() == "":
        st.error("Please enter some text to analyze.")
    else:
        # Preprocess input text
        processed_text = preprocess_text(input_text)

        # Transform input using TF-IDF
        input_vector = tfidf.transform([processed_text]).toarray()

        # Predict sentiment
        prediction = model.predict(input_vector)

        # Display result
        
        st.success(f"The predicted sentiment is: **{prediction[0]}**")
