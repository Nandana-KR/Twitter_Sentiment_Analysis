
**Twitter Sentiment Analysis Using Machine Learning**

This project builds a sentiment analysis model to classify Twitter messages as **Positive (1)** or **Negative (0)** using traditional machine learning techniques.
The workflow includes text preprocessing, TF-IDF vectorization, model training, evaluation, and saving the best-performing model.

---

 **Project Summary**

* Preprocesses and analyzes Twitter text data
* Converts tweets to numerical features using **TF-IDF** (unigrams + bigrams)
* Trains and compares three ML models:

  * **Naive Bayes**
  * **Logistic Regression**
  * **Linear SVM**
* Automatically selects the best model
* Performs evaluation using accuracy, classification report, and confusion matrix
* Saves the best model and vectorizer as `.pkl` files for reuse

---

**Best Model**

| Model               | Accuracy   |
| ------------------- | ---------- |
| Naive Bayes         | 95.12%     |
| Logistic Regression | 93.74%     |
| **Linear SVM**      | **96.31%** |

**Final model used:** LinearSVC
Saved as: `sentiment_model.pkl`
Vectorizer saved as: `tfidf_vectorizer.pkl`

---

**Dataset**

* Total tweets: **31,962**
* Columns:

  * `id` – Tweet ID
  * `label` – Sentiment (0 = Negative, 1 = Positive)
  * `tweet` – Tweet text


---

**Project Structure**

```
twitter_sentiment_project/
├── sentiment_analysis.py
├── twitter.csv
├── sentiment_model.pkl
├── tfidf_vectorizer.pkl
└── README.md
```

---

**Requirements**

Install required dependencies:

```
pip install pandas numpy seaborn matplotlib scikit-learn joblib
```

---

 **How to Run**

Run the training script:

```
python sentiment_analysis.py
```

The script will:

* Train all models
* Evaluate performance
* Identify the best model
* Save the model + vectorizer

---

**Predicting Sentiment on New Text**

```python
import joblib

model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

text = "I absolutely love this!"
vec = vectorizer.transform([text])
print(model.predict(vec))
```

Output:

* `1` → Positive
* `0` → Negative

---

**License**

This project is licensed under the **MIT License**.


