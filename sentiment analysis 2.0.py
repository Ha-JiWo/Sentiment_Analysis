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
# 定义HTML前端页面模板
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <!-- 适配移动端显示 -->
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <!-- 网页标题 -->
    <title>Movie Review Sentiment Analysis</title>
    <!-- 引入Chart.js图表库，用于绘制数据分布柱状图 -->
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <style>
        /* 全局样式：字体、边距、背景色 */
        body {
            font-family: Arial, sans-serif;
            margin: 40px;
            background: #f0f2f5;
        }
        /* 容器样式：居中、最大宽度、白色背景、阴影、圆角 */
        .container {
            max-width: 1200px;
            margin: auto;
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        /* 标题字体颜色 */
        h1, h2, h3 {
            color: #333;
        }
        /* 卡片样式：灰色背景、内边距、边框、圆角 */
        .card {
            background: #f9f9f9;
            padding: 15px;
            margin-bottom: 20px;
            border-radius: 6px;
            border: 1px solid #ddd;
        }
        /* 表格样式：宽度100%、合并边框、顶部边距 */
        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 10px;
        }
        /* 表格单元格样式：边框、内边距、左对齐 */
        th, td {
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }
        /* 表格表头背景色 */
        th {
            background-color: #eee;
        }
        /* 输入框样式：宽度100%、内边距、字体大小、边框、圆角 */
        textarea {
            width: 100%;
            padding: 10px;
            font-size: 14px;
            border-radius: 4px;
            border: 1px solid #ccc;
        }
        /* 按钮样式：蓝色背景、白色文字、无边框、圆角、鼠标悬浮效果 */
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
        /* 结果展示区域样式 */
        .result {
            margin-top: 20px;
            background: #e9ecef;
            padding: 15px;
            border-radius: 6px;
        }
        /* 置信度条目样式 */
        .confidence-item {
            margin-bottom: 10px;
        }
        /* 进度条背景 */
        .progress {
            background-color: #ddd;
            height: 8px;
            width: 100%;
            border-radius: 4px;
        }
        /* 进度条填充条 */
        .progress-bar {
            height: 8px;
            border-radius: 4px;
            background-color: #007bff;
        }
    </style>
</head>
<body>
<!-- 页面主容器 -->
<div class="container">
    <h1>Movie Review Sentiment Analysis</h1>
    <p>Model: TF-IDF + RidgeClassifier | Dataset: IMDB reviews (positive/negative/neutral)</p>

    <!-- 数据集分布模块 -->
    <div class="card">
        <h2>Dataset Distribution</h2>
        <!-- 图表画布 -->
        <canvas id="labelChart" style="max-height: 300px;"></canvas>
        <!-- 数据集统计表格 -->
        <table>
            <thead>
                <tr><th>Sentiment</th><th>Count</th><th>Percentage</th></tr>
            </thead>
            <tbody>
                <!-- Jinja2模板语法：循环渲染标签数据 -->
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

    <!-- 模型性能指标模块 -->
    <div class="card">
        <h2>Model Performance (Test Set)</h2>
        <p><strong>Accuracy:</strong> {{ "%.2f"|format(accuracy * 100) }}%</p>
        <p><strong>Macro F1:</strong> {{ "%.4f"|format(macro_f1) }}</p>
        <h3>Per-class Metrics</h3>
        <!-- 各类别精确率/召回率/F1表格 -->
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

    <!-- 在线预测模块 -->
    <div class="card">
        <h2>Predict Sentiment</h2>
        <!-- 影评输入框 -->
        <textarea id="reviewText" rows="3" placeholder="Enter an English movie review..."></textarea>
        <!-- 分析按钮 -->
        <button id="predictBtn">Analyze</button>
        <!-- 预测结果区域（默认隐藏） -->
        <div id="resultArea" class="result" style="display: none;">
            <h3>Prediction: <span id="predictionLabel"></span></h3>
            <h4>Confidence Scores:</h4>
            <div id="confidenceScores"></div>
        </div>
    </div>
</div>

<script>
    // 从后端获取标签、数量数据，转为JSON格式
    const labels = {{ labels | tojson }};
    const counts = {{ counts | tojson }};
    // 获取图表画布上下文
    const ctx = document.getElementById('labelChart').getContext('2d');
    // 渲染数据集分布柱状图
    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Number of reviews',
                data: labels.map(l => counts[l]),
                // 分别对应积极/消极/中性的颜色
                backgroundColor: ['#2ecc71', '#e74c3c', '#f39c12']
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true
        }
    });

    // 绑定按钮点击事件，执行预测请求
    document.getElementById('predictBtn').addEventListener('click', async function() {
        // 获取输入框内容并去除首尾空格
        const review = document.getElementById('reviewText').value.trim();
        // 空输入校验
        if (review === "") {
            alert("Please enter a review.");
            return;
        }
        // 显示结果区域
        const resultDiv = document.getElementById('resultArea');
        resultDiv.style.display = 'block';
        // 临时显示分析中
        document.getElementById('predictionLabel').innerHTML = 'Analyzing...';
        document.getElementById('confidenceScores').innerHTML = '';

        try {
            // 向后端/predict接口发送POST请求
            const response = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ review: review })
            });
            // 解析返回的JSON数据
            const data = await response.json();
            // 处理后端返回的错误
            if (data.error) {
                document.getElementById('predictionLabel').innerHTML = 'Error: ' + data.error;
                return;
            }
            // 显示预测结果
            document.getElementById('predictionLabel').innerHTML = data.sentiment;
            let html = '';
            // 遍历各类别置信度，计算百分比并渲染进度条
            for (const [cls, score] of Object.entries(data.scores)) {
                // Softmax计算概率百分比
                const prob = (Math.exp(score) / (Object.values(data.scores).reduce((a,b) => a + Math.exp(b), 0)) * 100).toFixed(1);
                // 根据类别设置不同颜色
                let barColor = '#007bff';
                if (cls === 'positive') barColor = '#2ecc71';
                if (cls === 'negative') barColor = '#e74c3c';
                if (cls === 'neutral') barColor = '#f39c12';
                // 拼接置信度展示HTML
                html += `
                    <div class="confidence-item">
                        <strong>${cls}</strong> <span style="float:right;">${prob}%</span>
                        <div class="progress"><div class="progress-bar" style="width: ${prob}%; background-color: ${barColor};"></div></div>
                        <small>raw score: ${score.toFixed(3)}</small>
                    </div>
                `;
            }
            // 将置信度内容插入页面
            document.getElementById('confidenceScores').innerHTML = html;
        } catch (err) {
            // 捕获请求异常
            document.getElementById('predictionLabel').innerHTML = 'Request failed: ' + err.message;
        }
    });
</script>
</body>
</html>
"""

# 首页路由：渲染可视化页面
@app.route('/')
def index():
    # 合法情感标签列表
    labels_list = valid_labels
    # 构建标签-数量字典，不存在的标签数量为0
    counts_dict = {label: label_counts.get(label, 0) for label in labels_list}
    # 数据集总样本数
    total_samples = len(df)
    # 构建模板所需的模型指标字典
    metrics_for_template = {}
    for lbl in labels_list:
        if lbl in class_metrics:
            metrics_for_template[lbl] = class_metrics[lbl]
        else:
            # 无数据时填充默认0值
            metrics_for_template[lbl] = {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0, 'support': 0}
    # 渲染HTML模板，传入后端数据
    return render_template_string(
        HTML_TEMPLATE,
        labels=labels_list,
        counts=counts_dict,
        total=total_samples,
        accuracy=accuracy,
        macro_f1=macro_f1,
        class_metrics=metrics_for_template
    )

# 预测接口路由：接收影评，返回情感预测结果
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 获取前端发送的JSON数据
        data = request.get_json()
        # 提取评论文本并去除空格
        review = data.get('review', '').strip()
        # 空文本校验
        if not review:
            return jsonify({'error': 'Empty review'}), 400
        # 模型预测情感标签
        pred_label = model.predict([review])[0]
        # 获取模型决策函数原始分数
        scores = model.decision_function([review])[0]
        # 构建 标签-分数 字典
        score_dict = dict(zip(model.classes_, scores))
        # 保证所有合法标签都有分数，缺失的设为极小值
        full_score_dict = {}
        for label in valid_labels:
            if label in score_dict:
                full_score_dict[label] = score_dict[label]
            else:
                full_score_dict[label] = -1e9
        # 返回JSON格式的预测结果和置信度
        return jsonify({'sentiment': pred_label, 'scores': full_score_dict})
    except Exception as e:
        # 捕获所有异常，返回错误信息
        return jsonify({'error': str(e)}), 500

# 主程序入口：启动Flask服务
if __name__ == '__main__':
    # 控制台打印模型训练完成信息
    print("\n" + "="*60)
    print("Model training completed. Starting web server...")
    print(f"Test accuracy: {accuracy:.4f}")
    print(f"Label distribution: {label_counts}")
    print("="*60 + "\n")
    # 启动Web服务，debug模式，本机所有IP可访问，端口5000
    app.run(debug=True, host='0.0.0.0', port=5000)
