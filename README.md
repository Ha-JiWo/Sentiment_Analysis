# Sentiment_Analysis
CS183 T25

## English Sentiment Analyzer

A lightweight sentiment analysis web application for English product reviews.  
It uses **TF‑IDF vectorization** and a **RidgeClassifier** with balanced class weights to classify reviews into `positive`, `negative`, or `neutral`.  
The model is trained on the IMDB dataset and served through a friendly web interface (Flask).


##  Features

- Machine learning based sentiment classification
- Handles three classes: positive, negative, neutral
- Clean web interface with:
  - Dataset distribution chart
  - Per‑class precision/recall/F1 metrics
  - Real‑time prediction with confidence scores (softmax‑like probability bars)
- Built with `scikit-learn`, `pandas`, `Flask`
- Automatically balances classes to improve neutral review detection

## Prerequisites
- Python 3.6 or higher

## Installation
1. **Clone or download** the project files.  
   Make sure the main script (e.g. `sentiment_app.py`) and the IMDB dataset (`IMDB Dataset.csv`) are in the same folder.

2. **Install required libraries**:

   ```bash
   pip install pandas numpy scikit-learn flask
   ```

## Usage

**Run the application**

In the terminal, navigate to the folder containing sentiment_app.py and execute:

```bash
python sentiment analysis 2.0.py
```

The script will:

-Load and filter the IMDB dataset (keeps only positive, negative, neutral labels)

-Split into training (80%) and test (20%) sets

-Train the TF‑IDF + RidgeClassifier pipeline (balanced class weights)

-Print accuracy and macro F1 score

-Start a Flask web server

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
