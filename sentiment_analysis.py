# -------------------------------------------
#   TWITTER SENTIMENT ANALYSIS 
# -------------------------------------------

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

import joblib

# -------------------------------------------
# 1. LOAD DATA
# -------------------------------------------

df = pd.read_csv("twitter.csv") 
print(df.head())
print(df.info())
print(df['label'].value_counts())

sns.countplot(x=df["label"])
plt.title("Sentiment Distribution")
plt.show()

# -------------------------------------------
# 2. FEATURE EXTRACTION (TF-IDF)
# No manual cleaning needed
# -------------------------------------------

vectorizer = TfidfVectorizer(
    stop_words="english",
    lowercase=True,
    ngram_range=(1,2),
    max_df=0.9,
    min_df=3
)

X = vectorizer.fit_transform(df["tweet"])
y = df["label"]

# -------------------------------------------
# 3. TRAIN / TEST SPLIT
# -------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------------------------
# 4. TRAIN MULTIPLE MODELS (BENCHMARKING)
# -------------------------------------------

models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
    "SVM": LinearSVC()
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    print("\n==============================")
    print("MODEL:", name)
    print("Accuracy:", acc)
    print("Classification Report:\n", classification_report(y_test, preds))

    cm = confusion_matrix(y_test, preds)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix - {name}")
    plt.show()

    results[name] = acc

# -------------------------------------------
# 5. SELECT BEST MODEL
# -------------------------------------------

best_model_name = max(results, key=results.get)
best_model = models[best_model_name]

print("\n===================================")
print("BEST MODEL:", best_model_name)
print("BEST ACCURACY:", results[best_model_name])
print("===================================")

# -------------------------------------------
# 6. ERROR ANALYSIS
# -------------------------------------------

preds = best_model.predict(X_test)
errors = df.iloc[y_test.index][preds != y_test]

print("\nSample Misclassified Tweets:")
print(errors.head(10))

# -------------------------------------------
# 7. SAVE THE BEST MODEL + VECTORIZER
# -------------------------------------------

joblib.dump(best_model, "sentiment_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

print("\nModel and vectorizer saved successfully!")
