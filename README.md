# Mobile Product Review Sentiment Analyzer

A live, interactive web application built with Streamlit that utilizes advanced machine learning to classify the sentiment of mobile product reviews. This project was developed as a Capstone Graduation Challenge for the TechCrush AI and Machine Learning Bootcamp.

🔗 **[Live Application Link]: https://mobile-review-sentiment-analyzer.streamlit.app/ **

---

## 🧠 Project Overview
Understanding customer feedback is vital for product optimization. This end-to-end application automates the process of sentiment analysis on an Amazon mobile product dataset. It processes raw unstructured text input and predicts whether a customer's review is **Negative**, **Neutral**, or **Positive**.

### Key Features:
* **Real-Time Analysis:** Users can type or paste any product review to immediately see the predicted sentiment.
* **Batch Analytics:** Supports CSV file uploads for processing hundreds of reviews simultaneously with real-time tracking.
* **Custom Decision Thresholding:** Implements customized metrics to optimize prediction confidence across multi-class outputs.
* **Elegant UI:** Built with a clean, dark-mode optimized interface utilizing Streamlit components.

---

## 🛠️ Tech Stack & Architecture

* **Frontend Dashboard:** Streamlit
* **Machine Learning Model:** XGBoost Classifier
* **Text Vectorization:** TF-IDF (Term Frequency-Inverse Document Frequency)
* **Natural Language Processing (NLP):** NLTK (Stopwords extraction & Tokenization)
* **Data Manipulation:** Pandas & NumPy
* **Model Serialization:** Joblib

### Project Structure:
```text
├── env/                        # Local Virtual Environment (Ignored via .gitignore)
├── app.py                      # Main Streamlit web application interface
├── requirements.txt            # Pre-compiled cloud server dependencies
├── .gitignore                  # Keeps heavy system files out of version control
├── README.md                   # Project documentation
└── [your_model_files].joblib   # Saved vectorizer and trained XGBoost model files
