import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import lightgbm as lgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
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
st.title("📝 License Text Recognition and Classification")
st.write("""
    **This app classifies open-source license texts (e.g., MIT, GPL) using machine learning techniques.**
""")

# Load predefined dataset
df = pd.read_csv('license_data.csv')

# Use columns to structure the layout
st.subheader("🔍 Dataset Preview")
st.dataframe(df, height=500)  # Display a larger scrollable dataset preview

# Preprocess the text data
df['clean_text'] = df['License Text'].apply(preprocess_text)

# Convert string labels to numerical labels
label_encoder = LabelEncoder()
df['Label'] = label_encoder.fit_transform(df['Label'])

# Split dataset
X = df['clean_text']
y = df['Label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Vectorization using TF-IDF
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Model selection
st.subheader("⚙️ Model Selection")
model_choice = st.selectbox("Choose a classification model:", 
                            ("Naive Bayes", "Random Forest", "SVM", 
                             "Logistic Regression", "Gradient Boosting", 
                             "XGBoost", "LightGBM"))

# Model selection logic
if model_choice == "Naive Bayes":
    model = MultinomialNB()
elif model_choice == "Random Forest":
    model = RandomForestClassifier()
elif model_choice == "SVM":
    model = SVC()
elif model_choice == "Logistic Regression":
    model = LogisticRegression(max_iter=1000)
elif model_choice == "Gradient Boosting":
    model = GradientBoostingClassifier()
elif model_choice == "XGBoost":
    model = XGBClassifier(eval_metric='mlogloss')
    X_train_vec = X_train_vec.toarray()  # Convert sparse matrix to dense
    X_test_vec = X_test_vec.toarray()    # Convert sparse matrix to dense
elif model_choice == "LightGBM":
    model = lgb.LGBMClassifier()

# Add a progress spinner during model training
try:
    with st.spinner("Training the model..."):
        model.fit(X_train_vec, y_train)
except ValueError as e:
    st.error(f"ValueError: {str(e)}")
    st.stop()  # Stop execution to prevent further errors

# Predict on the test set
y_pred = model.predict(X_test_vec)

# Convert numerical labels back to original string labels for display
y_test_labels = label_encoder.inverse_transform(y_test)
y_pred_labels = label_encoder.inverse_transform(y_pred)

# Evaluation metrics
accuracy = accuracy_score(y_test_labels, y_pred_labels) * 100  # Convert accuracy to percentage
precision = precision_score(y_test_labels, y_pred_labels, average='weighted')
recall = recall_score(y_test_labels, y_pred_labels, average='weighted')
f1 = f1_score(y_test_labels, y_pred_labels, average='weighted')

# Display evaluation results with enhanced UI
st.subheader("📊 Model Evaluation")
st.markdown(f"""
    **Accuracy**: `{accuracy:.2f}%`
    
    **Precision**: `{precision:.4f}`
    
    **Recall**: `{recall:.4f}`
    
    **F1 Score**: `{f1:.4f}`
""", unsafe_allow_html=True)

# Make predictions on new text input
st.subheader("🧪 Test License Classification")
user_input = st.text_area("Enter license text for classification:", height=150)

# Add a button to trigger the prediction
if st.button("🔍 Classify License"):
    if user_input:
        clean_input = preprocess_text(user_input)
        input_vec = vectorizer.transform([clean_input])
        prediction = model.predict(input_vec)
        prediction_label = label_encoder.inverse_transform(prediction)
        
        # Show the result with a colored markdown
        st.markdown(f"""
            <div style="background-color:#f0f4f7; padding: 10px; border-radius: 5px;">
            <h3 style="color: #4CAF50;">Predicted License Type: {prediction_label[0]}</h3>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.error("Please enter license text to classify.")

# Footer with instructions
st.markdown("""
    ---
    📝 **Instructions**: 
    1. View the dataset preview above.
    2. Choose a model for training and evaluation.
    3. Enter license text in the text area and press **Classify License**.
""")
