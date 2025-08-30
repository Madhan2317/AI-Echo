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

---

## 🌟 Future Improvements

- Add batch review input & summary chart (positive vs negative distribution)  
- Deploy on Streamlit Cloud / Heroku / Hugging Face Spaces  
- Add more advanced models (Logistic Regression, Transformers, BERT, etc.)  
- Include word clouds for most common positive/negative terms  

---

## 🤝 Contributing

Contributions are welcome!  
Feel free to open issues or submit pull requests to improve the app.  

---

## 📸 Screenshots

### Input and Prediction
![Input Review](<img width="1919" height="870" alt="image" src="https://github.com/user-attachments/assets/e489c7ef-5f84-46d9-a799-df2dabe7da0c" />
)

### Probability Distribution
![Probability Chart](<img width="1919" height="868" alt="image" src="https://github.com/user-attachments/assets/d4b307c5-60eb-4b4f-8fff-8d78b2885fbd" />
)

---

## 📜 License

This project is licensed under the MIT License.  

---

⚡ Built with ❤️ using Python, Scikit-Learn, and Streamlit


