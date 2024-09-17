import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import string
import nltk
from nltk.corpus import stopwords

# Download stopwords if necessary
nltk.download('stopwords')

# Function to clean and preprocess text
def preprocess_text(text):
    # Lowercasing
    text = text.lower()
    # Removing punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Tokenization
    tokens = text.split()
    # Stopword removal
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return ' '.join(tokens)

# App title and description
st.title("License Text Recognition and Classification")
st.write("""
    This app classifies open-source license texts (e.g., MIT, GPL) using machine learning techniques.
    """)

# Upload dataset
uploaded_file = st.file_uploader("Upload a dataset (CSV with 'text' and 'label' columns)", type="csv")

if uploaded_file:
    # Load dataset
    df = pd.read_csv(uploaded_file)
    
  
    # Display more rows from the dataset
    st.subheader("Dataset Preview")
    st.dataframe(df)  # This will display the entire dataset with scrolling

    # Preprocess the text data
    
    df['clean_text'] = df['License Text'].apply(preprocess_text)
    
    # Split dataset
    X = df['clean_text']
    y = df['Label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Vectorization using TF-IDF
    vectorizer = TfidfVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # Model selection
    model_choice = st.selectbox("Choose a classification model", ("Naive Bayes", "Random Forest", "SVM"))
    
    if model_choice == "Naive Bayes":
        model = MultinomialNB()
    # Add other model choices here...
    
    # Train the model
    model.fit(X_train_vec, y_train)
    
    # Predict on the test set
    y_pred = model.predict(X_test_vec)
    
    # Evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    # Display evaluation results
    st.subheader("Model Evaluation")
    st.write(f"Accuracy: {accuracy:.4f}")
    st.write(f"Precision: {precision:.4f}")
    st.write(f"Recall: {recall:.4f}")
    st.write(f"F1 Score: {f1:.4f}")
    
    # Make predictions on new text input
    st.subheader("Test License Classification")
    user_input = st.text_area("Enter license text for classification:")
    
    if user_input:
        clean_input = preprocess_text(user_input)
        input_vec = vectorizer.transform([clean_input])
        prediction = model.predict(input_vec)
        st.write(f"Predicted License Type: {prediction[0]}")
