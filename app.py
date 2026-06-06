import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import os
import pandas as pd
from datetime import datetime

# ==========================================
# 1. LOAD THE FILES
# ==========================================

model = joblib.load('winning_sentiment_model.pkl')
tfidf = joblib.load('tfidf_vectorizer.pkl')


nltk.download('stopwords')
nltk.download('wordnet')
# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
negative_words = {'no', 'not', 'never', 'neither', 'nor', 'hardly', 'barely', 'scarcely', 'but', 'however', 'although'}
stop_words = stop_words  - negative_words

# ==========================================
# THE TEXT CLEANING FUNCTION
# ==========================================
def clean_text(text):
    global stop_words

    # Lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r'http\S+', '', text)

    # Remove punctuation and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Tokenization
    words = text.split()

    # Remove stopwords
    words = [word for word in words if word not in stop_words]

    # Lemmatization
    words = [lemmatizer.lemmatize(word) for word in words]

    return ' '.join(words)

# ==========================================
# STREAMLIT USER INTERFACE (UI)
# ==========================================
st.title("📱 Mobile Product Review Sentiment Analyzer")
st.write("Developed by Group 23. Enter a customer review below to predict its sentiment automatically.")

# --- ADD THIS TO APP.PY TO CREATE A SIDEBAR ---
with st.sidebar:
    st.header("📋 Project Details")
    st.markdown("**Course:** AI & ML Bootcamp")
    st.markdown("**Project:** Customer Sentiment Analysis")
    st.markdown("**Dataset:** Amazon Unlocked Mobile (30k Sample)")
    
    st.divider() # Draws a clean separation line
    
    st.subheader("👥 Group 23 Teammates")
    st.write("- Priscilla")
    st.write("- Omar")
    st.write("- Hilary")

# --- BRAND TRACKING INPUTS ---
# Dropdown for popular brands, and add an "Other" option
brand_options = ["Apple", "Samsung", "Google", "Xiaomi", "Tecno", "Infinix", "Infinix", "Other"]
selected_brand = st.selectbox("Select the Mobile Phone Brand:", brand_options)

# If they select "Other", let them type it manually
if selected_brand == "Other":
    selected_brand = st.text_input("Please specify the brand name:")

# Create a visually distinct container for the input space
with st.container(border=True):
    st.subheader("📝 Review Input")
    user_review = st.text_area(
        "Type or paste your mobile review below:", 
        placeholder="Example: The battery life is amazing",
        height=100
    )
    
    # Center your button or give it an icon
    analyze_btn = st.button("🚀 Run Sentiment Analysis", use_container_width=True)


# ==========================================
# THE PREDICTION LOGIC
# ==========================================
if analyze_btn:
    if user_review.strip() != "":
        cleaned = clean_text(user_review)
        vectorized = tfidf.transform([cleaned])
        
        # 1. Get raw probability metrics from the model
        probabilities = model.predict_proba(vectorized)[0]
        
        # Extract individual probabilities
        prob_neg = probabilities[0]   # Probability for Class 0
        prob_neu = probabilities[1]   # Probability for Class 1
        prob_pos = probabilities[2]   # Probability for Class 2
        
        # 2. Check if the model is too uncertain between positive and negative
        # If the gap between positive and negative is tight, override to Neutral (Class 1)
        if abs(prob_pos - prob_neg) < 0.20:
            prediction = 1
        else:
            # Otherwise, fall back to the highest probability score class
            prediction = model.predict(vectorized)[0]
        
        st.divider()
        st.subheader("📊 Analysis Results")
        
        # Create two visual columns for the display
        col1, col2 = st.columns([1, 2])
        
        with col1:
            if prediction == 0:
                st.metric(label="Target Label", value="Class 0", delta="Negative", delta_color="inverse")
            elif prediction == 1:
                st.metric(label="Target Label", value="Class 1", delta="Neutral", delta_color="off")
            else:
                st.metric(label="Target Label", value="Class 2", delta="Positive", delta_color="normal")

        with col2:
            if prediction == 0:
                st.error(f"### 🔴 Negative Review for {selected_brand}\n\nThis customer is dissatisfied with their **{selected_brand}** device. Review flagged for technical or support intervention.")
            elif prediction == 1:
                st.info(f"### 🟡 Neutral Review for {selected_brand}\n\nThis is a mixed review for **{selected_brand}**. The customer experienced both highlights and drawbacks.")
            else:
                st.success(f"### 🟢 Positive Review for {selected_brand}\n\nThis customer is highly satisfied with their **{selected_brand}** device. Excellent feedback for product rating.")       

        
        # Display the certainty breakdown under the result
        st.caption(f"**Model Breakdown Confidence:** Negative: {prob_neg*100:.1f}% | Neutral: {prob_neu*100:.1f}% | Positive: {prob_pos*100:.1f}%")
       
       
        # ========================================================
        # CSV LOGGING
        # ========================================================
        log_file = "activity_log.csv"
        
        # Map the numeric predictions (0, 1, 2) back to readable text words
        sentiment_labels = {0: "Negative", 1: "Neutral", 2: "Positive"}
        
        # Package up the current session data into a dictionary
        new_data = {
            "Timestamp": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
            "Brand": [selected_brand],
            "Review": [user_review],
            "Predicted_Sentiment": [sentiment_labels[prediction]]
        }
        
        # Convert the dictionary into a neat Pandas DataFrame row
        new_df = pd.DataFrame(new_data)
        
        # Check if the file doesn't exist yet. If it's missing, create it with headers.
        if not os.path.isfile(log_file):
            new_df.to_csv(log_file, index=False)
        else:
            # If it already exists, append ('mode=a') the new row cleanly at the bottom without rewriting the header
            new_df.to_csv(log_file, mode='a', header=False, index=False)