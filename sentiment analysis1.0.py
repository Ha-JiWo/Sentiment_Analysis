import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score

# ============================================================
# 1. 读取 CSV，强制只取前两列，并过滤脏数据
# ============================================================
file_path = r'C:/Users/19293/Desktop/CS183/IMDB Dataset.csv'

try:
    # 尝试 UTF-8 读取，如果文件中有非英文字符可选用
    df = pd.read_csv(file_path, encoding='utf-8')
except UnicodeDecodeError:
    # UTF-8 失败则用 latin-1（兼容性更强）
    df = pd.read_csv(file_path, encoding='latin-1')
except FileNotFoundError:
    raise FileNotFoundError(f"❌ 文件未找到: {file_path}")

# 只保留前两列（text, label），忽略因多余逗号产生的 Unnamed 列
df = df.iloc[:, :2]
df.columns = ['text', 'label']

print(f"✅ 成功读取，原始行数: {len(df)}")
print("原始标签分布:")
print(df['label'].value_counts())

# ============================================================
# 2. 只保留我们需要的三类情感标签
# ============================================================
valid_labels = ['positive', 'negative', 'neutral']
df = df[df['label'].isin(valid_labels)]

print(f"✅ 过滤后行数: {len(df)}")
print("过滤后标签分布:")
print(df['label'].value_counts())

# ============================================================
# 3. 划分训练集和测试集（安全使用 stratify）
# ============================================================
# 检查是否有类别样本数少于2，若少于2则不能使用 stratify
min_class_count = df['label'].value_counts().min()
if min_class_count < 2:
    print(f"⚠️ 发现样本数少于2的类别，将不使用分层抽样 (stratify) 以避免报错。")
    stratify_param = None
else:
    stratify_param = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['label'],
    test_size=0.2, random_state=42,
    stratify=stratify_param
)

print(f"训练集大小: {len(X_train)}, 测试集大小: {len(X_test)}")

# ============================================================
# 4. 构建模型流水线（Ridge 分类器原生支持多分类）
# ============================================================
model = make_pipeline(
    TfidfVectorizer(
        max_features=5000,
        sublinear_tf=True,
        ngram_range=(1, 2),      # 包含单个词和双词组合，能捕捉"not good"等
        stop_words='english'     # 去掉常见英文停用词
    ),
    RidgeClassifier(alpha=1.0)
)

# ============================================================
# 5. 训练模型
# ============================================================
print("\n开始训练三分类模型（positive / neutral / negative）...")
model.fit(X_train, y_train)

# ============================================================
# 6. 评估模型
# ============================================================
y_pred = model.predict(X_test)

print("\n" + "="*50)
print("📊 模型评估报告（三分类）")
print("="*50)
print(classification_report(y_test, y_pred))
macro_f1 = f1_score(y_test, y_pred, average='macro')
print(f"Macro F1 Score: {macro_f1:.4f}")

# ============================================================
# 7. 预测新评论
# ============================================================
new_reviews = [
    "Terrible",
    "I don't know.",
    "This is just fine......",
    "An absolute masterpiece! I loved every second of it.",
    "It was a waste of time, completely boring.",
    "The movie was alright, nothing special.",
    "Not bad, but I probably wouldn't watch it again."
]

print("\n" + "="*50)
print("🔮 新评论预测结果")
print("="*50)
for review in new_reviews:
    pred = model.predict([review])[0]
    print(f"评论: {review:50s} → 预测情感: {pred}")

# ============================================================
# 8. （可选）保存模型以便后续使用
# ============================================================
# import joblib
# joblib.dump(model, 'sentiment_model.pkl')
# print("\n模型已保存为 sentiment_model.pkl")
