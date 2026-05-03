# CogniSense CX - ML Final Project Starter

This project is a starter setup for:

**CogniSense CX: Intelligent Complaint Intelligence & Customer Experience Analytics System**

## Project Structure

- `dataset/` raw and cleaned datasets
- `src/training/` data prep and model training scripts
- `src/app/` Flask localhost web app
- `models/` trained model files
- `outputs/charts/` generated plots
- `outputs/reports/` metrics reports
- `docs/` paper notes and documentation

## 1) Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 2) Download CFPB Dataset

Option A (manual):
- Download from `https://files.consumerfinance.gov/ccdb/complaints.csv.zip`
- Put extracted `complaints.csv` into `dataset/`

Option B (script):

```bash
python src/training/prepare_data.py
```

## 3) Train Baseline Model

```bash
python src/training/train_baseline.py
```

This now trains and compares two models:
- TF-IDF + Logistic Regression
- TF-IDF + Naive Bayes

Comparison output:
- `outputs/reports/model_comparison.txt`

Outputs:
- `models/issue_classifier.joblib`
- `models/label_encoder.joblib`
- `outputs/reports/classification_report.txt`
- `outputs/charts/confusion_matrix.png`

## 4) Run Localhost System

```bash
python src/app/app.py
```

This starts with Waitress (faster/stabler) if installed. To force Flask dev server:

```bash
set USE_WAITRESS=0
python src/app/app.py
```

Open: `http://127.0.0.1:5000`

Default login:
- username: `admin`
- password: `admin123`

## Added Project Features

- Sentiment signal on prediction results (`Positive/Neutral/Negative`)
- Urgency scoring (`Low/Medium/High` with score out of 100)
- Landing section pages (`/landing/analytics`, `/landing/models`, `/landing/insights`, `/landing/explore`)
- Optional multi-source preparation:
  - `python src/training/prepare_data.py --twitter-csv "path/to/twitter.csv"`
  - Generates `dataset/combined_text_sources.csv`

## Notes for Submission

- Midterm paper: use methodology and metrics from generated outputs.
- Finalterm system: run Flask app on localhost and demo upload/prediction/dashboard.
- Project folder: include `dataset`, `src`, `models`, `outputs`, and screenshots.
