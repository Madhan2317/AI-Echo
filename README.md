# 📝 Sentiment Analysis Dashboard

A simple yet powerful **Sentiment Analysis Dashboard** built with **Streamlit**, **TF-IDF Vectorizer**, and **Multinomial Naive Bayes**.  
This app predicts whether a product/service review is **Positive** or **Negative**, and displays prediction probabilities with an interactive chart.  

---

## 🚀 Features

- 🔍 Classifies reviews into Positive or Negative  
- 📊 Displays prediction probabilities with interactive Plotly bar chart  
- 🎨 Dark theme UI with modern card layout  
- ⚡ Built with Streamlit for fast deployment and interactivity  
- 📦 Easy to extend with other ML models (Logistic Regression, Random Forest, etc.)  

---

## 🧪 Usage

- Enter any product/service review in the text box.  
- Click **Predict Sentiment**.  
- The app will display:  
- ✅ Predicted sentiment (Positive/Negative)  
- 📊 Prediction probability distribution (interactive chart)  

---

## 📊 Model Training

- Dataset is cleaned and balanced using RandomOverSampler.  
- Text is vectorized with TF-IDF (bigrams, max_features=3000).  
- Model: Multinomial Naive Bayes (best suited for sparse text data).  
- Saved with joblib for reusability.  





