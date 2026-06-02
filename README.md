# Sentiment_Analysis
CS183 T25

# English Sentiment Analyzer (Rule‑Based Sentiment Analyzer)

A lightweight sentiment analysis tool for English product reviews, using a simple rule‑based approach with **positive/negative word lists** to classify text as `Positive`, `Negative`, or `Neutral`.

## ✨ Features

- Predefined positive and negative word sets
- Pure Python implementation, no third‑party libraries required
- Interactive command line interface – enter reviews one by one and see sentiment in real time

### Prerequisites
- Python 3.6 or higher

## Installation
**Install dependencies**:

```bash
pip install pandas numpy scikit-learn flask
```

## Usage

**Run the application**

In the terminal, navigate to the folder containing sentiment_app.py and execute:

```bash
python sentiment analysis 2.0.py
```

You will see output like:
```bash
 * Serving Flask app 'sentiment analysis 2.0'
 * Debug mode: on
WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.
 * Running on all addresses (0.0.0.0)
 * Running on http://127.0.0.1:5000
Press CTRL+C to quit
 * Restarting with watchdog (windowsapi)
Data size after filtering: 53396
Label distribution: {'positive': 24999, 'negative': 24999, 'neutral': 3398}
Training set size: 42716  Test set size: 10680

Training the model (with balanced class weights)...
Test Accuracy: 0.8896
Macro F1: 0.9195

============================================================
Model training completed. Starting web server...
Test accuracy: 0.8896
Label distribution: {'positive': 24999, 'negative': 24999, 'neutral': 3398}
============================================================
```
Open the web interface

Go to http://127.0.0.1:5000 or http://localhost:5000 in your browser.
