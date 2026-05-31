import pandas as pd
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score

# ============================================================
# 1. 读取并清洗数据（只保留 text 和 label）
# ============================================================
file_path = r'C:/Users/19293/Desktop/CS183/IMDB Dataset.csv'

try:
    df = pd.read_csv(file_path, encoding='utf-8')
except UnicodeDecodeError:
    df = pd.read_csv(file_path, encoding='latin-1')

# 只取前两列
df = df.iloc[:, :2]
df.columns = ['text', 'label']

# 只保留三类有效标签
valid_labels = ['positive', 'negative', 'neutral']
df = df[df['label'].isin(valid_labels)]

print(f"Data size after filtering: {len(df)}")
print("Label distribution:")
print(df['label'].value_counts())

# ============================================================
# 2. 划分训练/测试集（neutral 少于 2 条时不用 stratify）
# ============================================================
min_count = df['label'].value_counts().min()
stratify_param = df['label'] if min_count >= 2 else None

X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['label'],
    test_size=0.2, random_state=42,
    stratify=stratify_param
)

print(f"Training set size: {len(X_train)}  Test set size: {len(X_test)}")

# ============================================================
# 3. 构建流水线（加入 class_weight='balanced'）
# ============================================================
model = make_pipeline(
    TfidfVectorizer(
        max_features=5000,
        sublinear_tf=True,
        ngram_range=(1, 2),
        stop_words='english'
    ),
    RidgeClassifier(alpha=1.0, class_weight='balanced')
)

print("\nTraining the model (with balanced class weights)...")
model.fit(X_train, y_train)

# ============================================================
# 4. 基础评估
# ============================================================
y_pred = model.predict(X_test)
print("\nModel Evaluation Report")
print(classification_report(y_test, y_pred, zero_division=0))
print(f"Macro F1: {f1_score(y_test, y_pred, average='macro'):.4f}")

# ============================================================
# 5. 交互式评论情感分析
# ============================================================
print("\n" + "="*60)
print("Movie Review Sentiment Analyzer (type 'quit' or 'exit' to stop)")
print("="*60)

while True:
    user_input = input("\nEnter an English movie review: ").strip()
    if user_input.lower() in ['quit', 'exit', 'q']:
        print("Thank you for using the analyzer. Goodbye!")
        break
    if user_input == "":
        print("Warning: Empty input. Please enter a review.")
        continue

    # 预测情感与分值
    prediction = model.predict([user_input])[0]
    scores = model.decision_function([user_input])[0]
    score_details = dict(zip(model.classes_, scores))

    print(f"Predicted sentiment: {prediction}")
    print(f"Confidence scores per class: {score_details}")
