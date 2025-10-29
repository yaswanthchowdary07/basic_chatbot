import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# --- Load dataset ---
df = pd.read_csv("chat_bio.csv")

# --- Features and labels ---
X = df['text']
y = df['intent']

# --- Split data ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# --- TF-IDF Vectorization ---
vectorizer = TfidfVectorizer(ngram_range=(1,2))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# --- Train Logistic Regression Model ---
clf = LogisticRegression(max_iter=500)
clf.fit(X_train_vec, y_train)

# --- Predictions ---
y_pred = clf.predict(X_test_vec)

# --- Confusion Matrix ---
cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
print("Classification Report:\n", classification_report(y_test, y_pred))

# --- Plot Heatmap ---
plt.figure(figsize=(10,7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=clf.classes_, yticklabels=clf.classes_)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix Heatmap - Chatbot Intent Classification')
plt.show()
