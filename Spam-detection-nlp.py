
#Step 1: Read the File & Set Up Variables
#Goal: Load the data and check what it contains.

# Import necessary libraries
import pandas as pd

# Read the CSV file
df = pd.read_csv("spam.csv", encoding='ISO-8859-1')

# Look at the first few rows
print(df.head())

# See what columns exist
print(df.columns)

#Step 2: Clean the Data
#Goal: Remove useless columns and keep only what we need.

# Keep only the useful columns
df = df[['v1', 'v2']]

# Rename columns for clarity
df.columns = ['label', 'message']

# Drop missing values and reset index
df = df.dropna().reset_index(drop=True)

#Step 3: Convert Labels to Numbers
#Goal: Convert text labels into numeric form for ML models.

df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})

#Step 4: Add Message Length (for analysis)
#Goal: Analyze whether spam messages tend to be longer.

df['message_length'] = df['message'].apply(len)

print(df.head())
print(df.columns)

#Step 5: Plot Basic Graphs (Optional for Report)
#Goal: Visual understanding of the dataset.

import matplotlib.pyplot as plt
import seaborn as sns

# Count plot of ham vs spam
sns.countplot(x='label', data=df)
plt.title("Number of Ham vs Spam Messages")
plt.show()

# Message length distribution
sns.histplot(data=df, x='message_length', hue='label', bins=50, kde=True)
plt.title("Message Length Distribution")
plt.show()

#Step 6: Split Data into Train/Test
#Goal: Prepare data for training and evaluation.

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    df['message'],
    df['label_num'],
    test_size=0.2,
    random_state=42,
    stratify=df['label_num']
)

# Reset index for safe slicing later
X_train = X_train.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)

#Step 7: Convert Text to Numbers (TF-IDF)
#Goal: Convert text into numerical features.

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(stop_words='english', max_df=0.95)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

#Step 7A: Bag of Words Vectorization (NEW)
#Goal: Use simple word-frequency representation as baseline.

from sklearn.feature_extraction.text import CountVectorizer

bow_vectorizer = CountVectorizer(stop_words='english')

X_train_bow = bow_vectorizer.fit_transform(X_train)
X_test_bow = bow_vectorizer.transform(X_test)

#Step 7B: TF-IDF with N-grams (NEW)
#Goal: Capture multi-word spam phrases like “free entry”.

ngram_vectorizer = TfidfVectorizer(
    stop_words='english',
    ngram_range=(1, 3),
    max_df=0.95
)

X_train_ngram = ngram_vectorizer.fit_transform(X_train)
X_test_ngram = ngram_vectorizer.transform(X_test)

#Step 8: Train Models
#Goal: Train multiple ML models for comparison.

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Naive Bayes
nb_model = MultinomialNB()
nb_model.fit(X_train_tfidf, y_train)

# Logistic Regression
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train_tfidf, y_train)

# Random Forest
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train_tfidf, y_train)

#Step 8A: Handle Class Imbalance using SMOTE (NEW)

#Goal: Improve spam recall by balancing the dataset.

from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)

X_train_smote, y_train_smote = smote.fit_resample(
    X_train_tfidf, y_train
)

# Retrain Naive Bayes on balanced data
nb_smote = MultinomialNB()
nb_smote.fit(X_train_smote, y_train_smote)

#Step 9: Model Evaluation
#Goal: Evaluate models using precision, recall, and F1-score.

from sklearn.metrics import classification_report

# Predictions
nb_preds = nb_model.predict(X_test_tfidf)
lr_preds = lr_model.predict(X_test_tfidf)
rf_preds = rf_model.predict(X_test_tfidf)

print("Naive Bayes:\n", classification_report(y_test, nb_preds))
print("Logistic Regression:\n", classification_report(y_test, lr_preds))
print("Random Forest:\n", classification_report(y_test, rf_preds))

# Step 10: Confusion Matrix Visualization
# Goal: Visualize classification errors.

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

models_preds = {
    "Naive Bayes": nb_preds,
    "Logistic Regression": lr_preds,
    "Random Forest": rf_preds
}

for name, preds in models_preds.items():
    cm = confusion_matrix(y_test, preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Ham", "Spam"])
    disp.plot(cmap="Blues", values_format='d')
    plt.title(f"{name} Confusion Matrix")
    plt.show()

#Step 11: Threshold Tuning (Precision–Recall Tradeoff)
#Goal: Reduce false positives by tuning threshold.

probs = nb_model.predict_proba(X_test_tfidf)[:, 1]

thresholds = [0.3, 0.4, 0.5, 0.6]

from sklearn.metrics import classification_report

for t in thresholds:
    preds_thresh = (probs > t).astype(int)
    print(f"\nThreshold = {t}")
    print(classification_report(y_test, preds_thresh))

# Step 12: ROC–AUC Curve (NEW – Mandatory)
#Goal: Evaluate performance across all thresholds.

from sklearn.metrics import roc_curve, auc

fpr, tpr, _ = roc_curve(y_test, probs)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,4))
plt.plot(fpr, tpr, label=f"Naive Bayes (AUC = {roc_auc:.3f})")
plt.plot([0,1], [0,1], linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve – Spam Detection")
plt.legend()
plt.show()

print(f"ROC-AUC Score: {roc_auc:.4f}")

#Step 13: False Positive Analysis
#Goal: Understand why ham messages were misclassified as spam.

false_positives = X_test[(y_test == 0) & (nb_preds == 1)]

print("Sample False Positive Messages:")
print(false_positives.head(10))

#Step 14: Feature Importance (Logistic Regression)
#Goal: Identify words influencing predictions.

import numpy as np

feature_names = vectorizer.get_feature_names_out()
coefficients = lr_model.coef_[0]

word_importance = list(zip(feature_names, coefficients))

top_spam = sorted(word_importance, key=lambda x: x[1], reverse=True)[:20]
top_ham = sorted(word_importance, key=lambda x: x[1])[:20]

print("📢 Top spam-indicating words:")
for word, coef in top_spam:
    print(f"{word:<15} -> {coef:.3f}")

print("\n💬 Top ham-indicating words:")
for word, coef in top_ham:
    print(f"{word:<15} -> {coef:.3f}")

#Step 15: Word Clouds
#Goal: Visualize common words in spam and ham messages.

from wordcloud import WordCloud

spam_words = ' '.join(df[df['label_num'] == 1]['message'])
ham_words = ' '.join(df[df['label_num'] == 0]['message'])

plt.figure(figsize=(10,5))
plt.title("Spam Word Cloud")
plt.imshow(WordCloud(background_color='white').generate(spam_words))
plt.axis("off")
plt.show()

plt.figure(figsize=(10,5))
plt.title("Ham Word Cloud")
plt.imshow(WordCloud(background_color='white').generate(ham_words))
plt.axis("off")
plt.show()

#Step 16: Save Model & Vectorizer
#Goal: Persist trained artifacts for deployment.

import joblib

joblib.dump(nb_model, 'spam_classifier_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')

