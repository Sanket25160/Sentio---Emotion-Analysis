# 🧠 Sentio

**AI-powered emotion analysis from text using Natural Language Processing and Machine Learning.**

Live at: https://sanket25160-sentio---emotion-analysis-app-5yaemz.streamlit.app/

Sentio is an NLP-based application that analyzes text and predicts the emotion expressed in it. The project uses **text preprocessing, Bag of Words (BoW), and Logistic Regression** for emotion classification.

## 🚀 Features

- 📝 Analyze emotions from text
- 🧹 Automatic text cleaning and preprocessing
- 🔤 Bag of Words feature extraction
- 🤖 Logistic Regression classification
- 📊 Emotion probability visualization
- 🔍 View the cleaned text
- 🌐 Simple and interactive Streamlit interface

## 🛠️ Technologies Used

- **Python**
- **Pandas** — Data handling
- **NLTK** — Text preprocessing
- **Scikit-learn** — Feature extraction and machine learning
- **Joblib** — Model serialization
- **Streamlit** — Web application

## 🧠 Machine Learning Pipeline

```text
Input Text
    ↓
Lowercase Conversion
    ↓
Punctuation Removal
    ↓
Number Removal
    ↓
Stopword Removal
    ↓
Bag of Words (CountVectorizer)
    ↓
Logistic Regression
    ↓
Predicted Emotion
```

## 🧹 Text Preprocessing

Sentio preprocesses the input text before passing it to the machine learning model.

The following steps are performed:

1. Convert text to lowercase
2. Remove punctuation
3. Remove numbers
4. Remove English stopwords

### Example

```text
Original:
"I am REALLY happy today!!!"

After cleaning:
"really happy today"
```

## 🤖 Machine Learning Model

Sentio uses:

**Logistic Regression + Bag of Words**

`CountVectorizer` converts the cleaned text into numerical feature vectors based on word occurrences.

These vectors are then passed to the trained Logistic Regression model to predict the emotion.

## 📂 Project Structure

```text
Sentio/
│
├── app.py
├── train.txt
│
├── bow_vectorizer.pkl
├── sentiment_model.pkl
├── emotion_mapping.pkl
│
├── requirements.txt
└── README.md
```

| File | Description |
|---|---|
| `app.py` | Streamlit application |
| `train.txt` | Training dataset |
| `bow_vectorizer.pkl` | Saved Bag of Words vectorizer |
| `sentiment_model.pkl` | Trained Logistic Regression model |
| `emotion_mapping.pkl` | Emotion-to-number mapping |
| `requirements.txt` | Required Python packages |
| `README.md` | Project documentation |

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/Sentio.git
```

### 2. Navigate to the project

```bash
cd Sentio
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
streamlit run app.py
```

The Sentio application will open in your browser.

## 📦 Requirements

```text
streamlit
pandas
scikit-learn
nltk
joblib
```

## 🖥️ How It Works

The user enters a sentence into the Sentio interface.

The application then:

1. Cleans the input text.
2. Converts it into Bag of Words features.
3. Passes the features to the Logistic Regression model.
4. Predicts the emotion.
5. Displays the predicted emotion and probability distribution.

## 📊 Example

**Input:**

```text
I am extremely happy and excited today!
```

**Output:**

```text
Predicted Emotion: JOY
Confidence: XX.XX%
```

The exact prediction depends on the trained model and dataset.

## 🔮 Future Improvements

- Compare different machine learning algorithms
- Experiment with TF-IDF features
- Perform hyperparameter tuning
- Improve text preprocessing
- Add emoji and special-character handling
- Add prediction history
- Deploy Sentio online
- Experiment with deep learning and transformer-based models

## 🎯 Learning Objectives

This project demonstrates the fundamental workflow of an NLP classification project:

- Data loading and preprocessing
- Text cleaning
- Feature extraction
- Train-test splitting
- Machine learning model training
- Model evaluation
- Model serialization
- Building an interactive ML application with Streamlit

## 👨‍💻 Author

**Sanket Singh**

B.Tech CSE | AI/ML Enthusiast

---

⭐ If you found Sentio useful, consider giving the repository a star!
