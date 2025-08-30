import streamlit as st
import joblib
import re
import pandas as pd
import plotly.express as px

# ---------------------------
# Load Model and Vectorizer
# ---------------------------
# Make sure you have trained MultinomialNB on your TF-IDF features
# joblib.dump(nb_model, "nb_model.joblib")
# joblib.dump(tfidf_vectorizer, "tfidf_vectorizer.joblib")

nb_model = joblib.load("multinomial_naive_bayes_model.pkl")
tfidf_vectorizer = joblib.load("tfidf_vectorizer.pkl")

# ---------------------------
# Text Preprocessing
# ---------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(page_title="Sentiment Analyzer", page_icon="📝", layout="wide")

# Custom CSS for dark theme
st.markdown("""
<style>
.stApp {
    background-color: #121212;
    color: #f0f0f0;
    font-family: 'Segoe UI', sans-serif;
}
.header {
    font-size: 42px;
    font-weight: bold;
    color: #1DB954;
    text-align: center;
}
.subheader {
    font-size: 18px;
    color: #B0B0B0;
    text-align: center;
    margin-bottom: 20px;
}
.card {
    background-color: #1E1E1E;
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.5);
}
.stTextArea>div>div>textarea {
    background-color: #2C2C2C;
    color: #f0f0f0;
}
.stButton>button {
    background-color: #1DB954;
    color: white;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="header">📝 Sentiment Analysis Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="subheader">Predict whether a review is Positive or Negative</div>', unsafe_allow_html=True)

# Sidebar
st.sidebar.header("Instructions")
st.sidebar.write("""
- Enter your product/service review in the box.
- Click **Predict Sentiment**.
- View the predicted sentiment and probability distribution.
- Positive predictions are shown in **green**, negative in **red**.
""")

# Input Section
with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    user_input = st.text_area("Enter Your Review Here:", "", height=150)
    st.markdown('</div>', unsafe_allow_html=True)

# Prediction
if st.button("Predict Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter a review to predict!")
    else:
        # Clean and transform
        cleaned_input = clean_text(user_input)
        input_tfidf = tfidf_vectorizer.transform([cleaned_input])

        # Predict
        prediction = nb_model.predict(input_tfidf)[0]
        prediction_proba = nb_model.predict_proba(input_tfidf)[0]

        # Display results
        with st.container():
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Predicted Sentiment")
            if prediction == "positive":
                st.success(f"**{prediction.capitalize()}**")
            else:
                st.error(f"**{prediction.capitalize()}**")

            # Probability chart
            proba_df = pd.DataFrame({
                "Sentiment": nb_model.classes_,
                "Probability": prediction_proba
            })

            fig = px.bar(proba_df, x="Sentiment", y="Probability",
                         text=proba_df["Probability"].apply(lambda x: f"{x*100:.1f}%"),
                         color="Sentiment",
                         color_discrete_map={"positive":"#1DB954", "negative":"#FF3B30"},
                         labels={"Probability":"Probability"},
                         title="Prediction Probability Distribution")
            fig.update_layout(
                plot_bgcolor='#121212',
                paper_bgcolor='#121212',
                font_color='#f0f0f0',
                title_x=0.5
            )
            fig.update_yaxes(range=[0,1])
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
