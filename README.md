# **Sentiment Analysis using DVC, Streamlit, and MLOps**

This project implements a sentiment analysis pipeline using machine learning, leveraging **DVC (Data Version Control)**, **Streamlit**, and **MLOps** principles to ensure reproducibility, scalability, and ease of testing. It focuses on classifying text data into six emotions: **Anger**, **Fear**, **Joy**, **Love**, **Sadness**, and **Surprise**.

---

## **Project Structure**

```
.
├── data
│   ├── preprocess
│   ├── raw
│   └── raw.dvc
├── models
│   ├── lemma.pkl
│   ├── model.pkl
│   └── tfidf.pkl
├── notebook
├── src
│   ├── __init__.py
│   ├── evaluate.py
│   ├── main.py         # Streamlit app
│   ├── preprocess.py
│   └── train.py
├── venv
├── .dvc
├── dvc.yaml
├── params.yaml
├── dvc.lock
├── README.md
└── requirements.txt
```

---

## **Dataset**

The dataset consists of English Twitter messages categorized into six emotions. The data underwent preprocessing steps like **lemmatization**, **stopword removal**, and **handling class imbalance using SMOTE (Synthetic Minority Over-sampling Technique)**.

### **Source**
The dataset was collected using hashtags for each emotion and preprocessed according to research guidelines.

---

## **Pipeline Stages**

This project employs **DVC** to orchestrate the following pipeline stages:

1. **Preprocessing**:  
   - Cleaned text data.
   - Tokenization and **lemmatization** using **NLTK**.  
   - Applied **SMOTE** to address class imbalance.

2. **Feature Engineering**:  
   - Converted text data into numerical vectors using **TF-IDF** (Term Frequency-Inverse Document Frequency).

3. **Model Training**:  
   - Trained a **SVC** for emotion classification.  
   - Saved the model, TF-IDF, and lemmatization objects using **Pickle**.

4. **Model Evaluation**:  
   - Evaluated model performance using accuracy, precision, recall, and F1-score.

5. **Streamlit App**:  
   - Developed a **Streamlit UI** for user-friendly testing of the model.

6. **Version Control**:  
   - Used **GitHub** for code and **DagsHub** for data and model versioning.

---

## **Model Evaluation**

### **Classification Report**

| Emotion      | Precision | Recall | F1-Score | Support |
|--------------|-----------|--------|----------|---------|
| **Anger**    | 0.91      | 0.82   | 0.86     | 1241    |
| **Fear**     | 0.82      | 0.82   | 0.82     | 1160    |
| **Joy**      | 0.70      | 0.83   | 0.76     | 1191    |
| **Love**     | 0.90      | 0.89   | 0.90     | 1234    |
| **Sadness**  | 0.84      | 0.77   | 0.80     | 1235    |
| **Surprise** | 0.95      | 0.96   | 0.96     | 1208    |

#### **Overall Metrics**:
- **Accuracy**: 0.85  
- **Macro Avg**:  
  - Precision: 0.85  
  - Recall: 0.85  
  - F1-Score: 0.85  
- **Weighted Avg**:  
  - Precision: 0.85  
  - Recall: 0.85  
  - F1-Score: 0.85  

---

## **Technologies and Tools**

- **Python**: For data preprocessing, model training, and evaluation.
- **DVC**: For pipeline orchestration and data version control.
- **GitHub**: For version control of the project codebase.
- **DagsHub**: For storing and tracking datasets, models, and pipelines.
- **Streamlit**: For creating an interactive UI for model testing.
- **NLTK**: For lemmatization and stopword removal.
- **Scikit-learn**: For feature engineering, modeling, and evaluation.
- **SMOTE**: For addressing class imbalance.
- **TF-IDF**: For vectorizing textual data.
- **Pickle**: For saving and loading model objects.

---

## **Streamlit UI**

The **Streamlit app** provides an intuitive interface to test the sentiment analysis model.  
### **Features**:
- Input a custom sentence and get its predicted emotion.  
- Displays probability scores for each emotion category.  

Run the app locally using:
```bash
streamlit run src/main.py
```

---

## **How to Run the Project**

1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd <repository_name>
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up DVC for pipeline execution:
   ```bash
   dvc pull   # Pull data and model files from remote storage
   dvc repro  # Reproduce the pipeline
   ```

4. Run the Streamlit app:
   ```bash
   streamlit run src/main.py
   ```

---

## **Version Control and Collaboration**

- **GitHub**:  
  The codebase is version-controlled using Git, ensuring collaboration and version history.  
- **DagsHub**:  
  Tracks datasets, pipeline metadata, and model versions seamlessly integrated with DVC.

---

## **Future Enhancements**

1. **Multilingual Support**: Extend support for non-English datasets.  
2. **Real-Time Inference**: Integrate the app with a REST API for real-time predictions.  
3. **Fine-Grained Emotions**: Include additional emotion categories for more granular predictions.  
4. **Explainable AI**: Provide interpretability for predictions using frameworks like SHAP.

