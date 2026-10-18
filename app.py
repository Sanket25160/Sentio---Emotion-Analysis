import streamlit as st
import joblib
import string
import nltk
import pandas as pd

from nltk.corpus import stopwords


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Emotion Analysis",
    page_icon="🧠",
    layout="centered"
)


# --------------------------------------------------
# Load NLTK Stopwords
# --------------------------------------------------

nltk.download("stopwords", quiet=True)

stop_words = set(stopwords.words("english"))


# --------------------------------------------------
# Load Model Files
# --------------------------------------------------

bow_vectorizer = joblib.load("bow_vectorizer.pkl")
model = joblib.load("sentiment_model.pkl")
emotion_mapping = joblib.load("emotion_mapping.pkl")


# Reverse emotion mapping

emotion_labels = {
    value: key
    for key, value in emotion_mapping.items()
}


# --------------------------------------------------
# Text Cleaning
# --------------------------------------------------

def clean_text(text):

    # Convert to lowercase
    text = text.lower()

    # Remove punctuation
    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )

    # Remove numbers
    text = "".join(
        char for char in text
        if not char.isdigit()
    )

    # Remove stopwords
    words = text.split()

    cleaned_words = []

    for word in words:

        if word not in stop_words:
            cleaned_words.append(word)

    return " ".join(cleaned_words)


# --------------------------------------------------
# UI
# --------------------------------------------------

st.title("🧠 Sentio")

st.write(
    "Enter a sentence and the model will predict "
    "the emotion expressed in the text."
)

st.divider()


# --------------------------------------------------
# Text Input
# --------------------------------------------------

text = st.text_area(
    "Enter your text",
    placeholder="Example: I am feeling really happy today!",
    height=150
)


# --------------------------------------------------
# Analyze Button
# --------------------------------------------------

if st.button(
    "Analyze Emotion",
    type="primary",
    use_container_width=True
):

    if text.strip() == "":
        st.warning("Please enter some text.")

    else:

        # Clean text
        cleaned_text = clean_text(text)

        # Convert text into BoW
        text_vector = bow_vectorizer.transform(
            [cleaned_text]
        )

        # Predict emotion
        prediction = model.predict(
            text_vector
        )[0]

        emotion = emotion_labels[prediction]

        # Get probabilities
        probabilities = model.predict_proba(
            text_vector
        )[0]

        probability_data = {
            emotion_labels[i]: probabilities[i]
            for i in range(len(probabilities))
        }

        # Highest probability
        confidence = probabilities[prediction] * 100


        # --------------------------------------------------
        # Result
        # --------------------------------------------------

        st.subheader("Prediction")

        st.success(
            f"**Emotion: {emotion.upper()}**"
        )

        st.write(
            f"Confidence: **{confidence:.2f}%**"
        )


        # --------------------------------------------------
        # Probability Chart
        # --------------------------------------------------

        st.subheader("Emotion Probabilities")

        probability_df = pd.DataFrame(
            {
                "Emotion": probability_data.keys(),
                "Probability": [
                    value * 100
                    for value in probability_data.values()
                ]
            }
        )

        probability_df = probability_df.sort_values(
            "Probability",
            ascending=False
        )

        st.bar_chart(
            probability_df.set_index("Emotion")
        )


        # --------------------------------------------------
        # Cleaned Text
        # --------------------------------------------------

        with st.expander("View cleaned text"):

            st.write(cleaned_text)


# --------------------------------------------------
# Footer
# --------------------------------------------------

st.divider()

st.caption(
    "Model: Logistic Regression | Features: Bag of Words"
)