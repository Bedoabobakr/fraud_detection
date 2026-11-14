# FraudGuard — Mobile Money Fraud Detection

A cost-aware machine-learning fraud detector built on the [PaySim dataset](https://www.kaggle.com/datasets/ealaxi/paysim1). Classroom demo, **not** a production fraud prevention service.

## Features
- EDA and feature engineering for balance-change anomalies
- Stratified sampling and baseline vs three machine-learning models
- Cross-validation, tuning, PR-AUC, ROC-AUC, precision/recall, confusion matrix
- Training-only threshold selection by business cost
- Streamlit single-transaction form, CSV scoring, dashboard, threshold/cost analysis

## How to run
1. Download the PaySim CSV from Kaggle and place it alongside `fraud_detection_ready.ipynb` (or install `kagglehub`).
2. Install notebook dependencies: `pip install numpy pandas matplotlib seaborn scikit-learn scipy joblib kagglehub streamlit`.
3. Open `fraud_detection_ready.ipynb` and choose **Restart & Run All**. It generates `fraud_pipeline.joblib`, `model_meta.json`, `sample_transactions.csv` and app files.
4. Run `streamlit run app.py` from this folder.
5. Upload `sample_transactions.csv` in the Batch scoring tab for an interactive threshold experiment.

## Results
**Fill this table with actual notebook output after training. Do not invent metrics.**

| Model | CV PR-AUC | Test PR-AUC | Precision | Recall | Total assumed cost |
|---|---:|---:|---:|---:|---:|
| Baseline | pending | pending | pending | pending | pending |
| Logistic Regression | pending | pending | pending | pending | pending |
| Random Forest | pending | pending | pending | pending | pending |
| HistGradientBoosting | pending | pending | pending | pending | pending |

## Deploy on Streamlit Community Cloud
Push the code, requirements, `fraud_features.py`, and **trained** `fraud_pipeline.joblib` + `model_meta.json` to GitHub. Do not upload the PaySim dataset. The model file may exceed GitHub limits; if so use Git LFS or an alternative model hosting/download mechanism. Deploy `app.py` from the repository on Streamlit Community Cloud and test the public URL. No live deployment has been performed here.

## Caveats
PaySim is synthetic, model scores are not calibrated probabilities, account balances may be post-transaction and unavailable at decision time, and business costs are assumptions. Threshold experimentation on labeled test data is illustrative only; choose deployment thresholds on validation data.

## Demo screenshot / link
Add after running and deploying the app.
