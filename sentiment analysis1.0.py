import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score

# 1. Read CSV file (handling encoding issues)
file_path = r'C:/Users/19293/Desktop/CS183/IMDB Dataset.csv'

try:
    # Try UTF-8 first
    df = pd.read_csv(file_path, encoding='utf-8')
    print("Read with UTF-8 encoding")
except UnicodeDecodeError:
    # Fallback to latin-1 if UTF-8 fails
    df = pd.read_csv(file_path, encoding='latin-1')
    print("UTF-8 failed, read with latin-1 encoding.")

# 2. Check column names (adjust according to actual data)
print("Column names in the CSV:", df.columns.tolist())
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['label'], test_size=0.2, random_state=42
)

# 3. Build model pipeline
model = make_pipeline(
    TfidfVectorizer(max_features=5000, sublinear_tf=True),
    RidgeClassifier(alpha=1.0)
)

# 4. Train the model
print(" Training the model... Santa's little helper is working! ")
model.fit(X_train, y_train)

# 5. Evaluate the model
y_pred = model.predict(X_test)
print(" Model evaluation report ")
print(classification_report(y_test, y_pred))
print(f" Macro F1 Score: {f1_score(y_test, y_pred, average='macro'):.4f}")

# 6. Predict on new reviews
new_reviews = [
    "Terrible",
    "OK",
    "This is just fine......"
]
print("\n️  Predicting sentiment for new reviews (Christmas edition)  ️")
predictions = model.predict(new_reviews)
for review, label in zip(new_reviews, predictions):
    print(f" Review: {review}")
    print(f" Predicted sentiment: {label}")
    print('-' * 30)
