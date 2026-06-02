import os
import pandas as pd
import numpy as np
from flask import Flask, render_template_string, request, jsonify
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, accuracy_score
app = Flask(__name__)
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
label_counts = df['label'].value_counts().to_dict()
print("Label distribution:", label_counts)

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
accuracy = accuracy_score(y_test, y_pred)
macro_f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
class_report = classification_report(y_test, y_pred, zero_division=0, output_dict=True)
class_metrics = {}
for label in valid_labels:
    if label in class_report:
        class_metrics[label] = class_report[label]
    else:
        class_metrics[label] = {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0, 'support': 0}

print(f"Test Accuracy: {accuracy:.4f}")
print(f"Macro F1: {macro_f1:.4f}")
# ============================================================
# 5. Flask
# ============================================================
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Movie Review Sentiment Analysis</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 40px;
            background: #f0f2f5;
        }
        .container {
            max-width: 1200px;
            margin: auto;
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        h1, h2, h3 {
            color: #333;
        }
        .card {
            background: #f9f9f9;
            padding: 15px;
            margin-bottom: 20px;
            border-radius: 6px;
            border: 1px solid #ddd;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 10px;
        }
        th, td {
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }
        th {
            background-color: #eee;
        }
        textarea {
            width: 100%;
            padding: 10px;
            font-size: 14px;
            border-radius: 4px;
            border: 1px solid #ccc;
        }
        button {
            background-color: #007bff;
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 4px;
            cursor: pointer;
            margin-top: 10px;
        }
        button:hover {
            background-color: #0056b3;
        }
        .result {
            margin-top: 20px;
            background: #e9ecef;
            padding: 15px;
            border-radius: 6px;
        }
        .confidence-item {
            margin-bottom: 10px;
        }
        .progress {
            background-color: #ddd;
            height: 8px;
            width: 100%;
            border-radius: 4px;
        }
        .progress-bar {
            height: 8px;
            border-radius: 4px;
            background-color: #007bff;
        }
    </style>
</head>
<body>
<div class="container">
    <h1>Movie Review Sentiment Analysis</h1>
    <p>Model: TF-IDF + RidgeClassifier | Dataset: IMDB reviews (positive/negative/neutral)</p>

    <div class="card">
        <h2>Dataset Distribution</h2>
        <canvas id="labelChart" style="max-height: 300px;"></canvas>
        <table>
            <thead>
                <tr><th>Sentiment</th><th>Count</th><th>Percentage</th></tr>
            </thead>
            <tbody>
                {% for label in labels %}
                <tr>
                    <td>{{ label }}</td>
                    <td>{{ counts[label] }}</td>
                    <td>{{ "%.1f"|format(counts[label] / total * 100) }}%</td>
                </tr>
                {% endfor %}
            </tbody>
        </table>
        <p><strong>Total samples:</strong> {{ total }}</p>
    </div>

    <div class="card">
        <h2>Model Performance (Test Set)</h2>
        <p><strong>Accuracy:</strong> {{ "%.2f"|format(accuracy * 100) }}%</p>
        <p><strong>Macro F1:</strong> {{ "%.4f"|format(macro_f1) }}</p>
        <h3>Per-class Metrics</h3>
        <table>
            <thead>
                <tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1-score</th><th>Support</th></tr>
            </thead>
            <tbody>
                {% for label in labels %}
                <tr>
                    <td>{{ label }}</td>
                    <td>{{ "%.3f"|format(class_metrics[label].precision) }}</td>
                    <td>{{ "%.3f"|format(class_metrics[label].recall) }}</td>
                    <td>{{ "%.3f"|format(class_metrics[label]['f1-score']) }}</td>
                    <td>{{ class_metrics[label].support }}</td>
                </tr>
                {% endfor %}
            </tbody>
        </table>
    </div>

    <div class="card">
        <h2>Predict Sentiment</h2>
        <textarea id="reviewText" rows="3" placeholder="Enter an English movie review..."></textarea>
        <button id="predictBtn">Analyze</button>
        <div id="resultArea" class="result" style="display: none;">
            <h3>Prediction: <span id="predictionLabel"></span></h3>
            <h4>Confidence Scores:</h4>
            <div id="confidenceScores"></div>
        </div>
    </div>
</div>

<script>
    const labels = {{ labels | tojson }};
    const counts = {{ counts | tojson }};
    const ctx = document.getElementById('labelChart').getContext('2d');
    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Number of reviews',
                data: labels.map(l => counts[l]),
                backgroundColor: ['#2ecc71', '#e74c3c', '#f39c12']
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true
        }
    });

    document.getElementById('predictBtn').addEventListener('click', async function() {
        const review = document.getElementById('reviewText').value.trim();
        if (review === "") {
            alert("Please enter a review.");
            return;
        }
        const resultDiv = document.getElementById('resultArea');
        resultDiv.style.display = 'block';
        document.getElementById('predictionLabel').innerHTML = 'Analyzing...';
        document.getElementById('confidenceScores').innerHTML = '';

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ review: review })
            });
            const data = await response.json();
            if (data.error) {
                document.getElementById('predictionLabel').innerHTML = 'Error: ' + data.error;
                return;
            }
            document.getElementById('predictionLabel').innerHTML = data.sentiment;
            let html = '';
            for (const [cls, score] of Object.entries(data.scores)) {
                const prob = (Math.exp(score) / (Object.values(data.scores).reduce((a,b) => a + Math.exp(b), 0)) * 100).toFixed(1);
                let barColor = '#007bff';
                if (cls === 'positive') barColor = '#2ecc71';
                if (cls === 'negative') barColor = '#e74c3c';
                if (cls === 'neutral') barColor = '#f39c12';
                html += `
                    <div class="confidence-item">
                        <strong>${cls}</strong> <span style="float:right;">${prob}%</span>
                        <div class="progress"><div class="progress-bar" style="width: ${prob}%; background-color: ${barColor};"></div></div>
                        <small>raw score: ${score.toFixed(3)}</small>
                    </div>
                `;
            }
            document.getElementById('confidenceScores').innerHTML = html;
        } catch (err) {
            document.getElementById('predictionLabel').innerHTML = 'Request failed: ' + err.message;
        }
    });
</script>
</body>
</html>
"""

@app.route('/')
def index():
    labels_list = valid_labels
    counts_dict = {label: label_counts.get(label, 0) for label in labels_list}
    total_samples = len(df)
    metrics_for_template = {}
    for lbl in labels_list:
        if lbl in class_metrics:
            metrics_for_template[lbl] = class_metrics[lbl]
        else:
            metrics_for_template[lbl] = {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0, 'support': 0}
    return render_template_string(
        HTML_TEMPLATE,
        labels=labels_list,
        counts=counts_dict,
        total=total_samples,
        accuracy=accuracy,
        macro_f1=macro_f1,
        class_metrics=metrics_for_template
    )

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        review = data.get('review', '').strip()
        if not review:
            return jsonify({'error': 'Empty review'}), 400
        pred_label = model.predict([review])[0]
        scores = model.decision_function([review])[0]
        score_dict = dict(zip(model.classes_, scores))
        full_score_dict = {}
        for label in valid_labels:
            if label in score_dict:
                full_score_dict[label] = score_dict[label]
            else:
                full_score_dict[label] = -1e9
        return jsonify({'sentiment': pred_label, 'scores': full_score_dict})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("\n" + "="*60)
    print("Model training completed. Starting web server...")
    print(f"Test accuracy: {accuracy:.4f}")
    print(f"Label distribution: {label_counts}")
    print("="*60 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000)
