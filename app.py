import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from fraud_features import RAW_COLS   # needed so joblib can unpickle the feature function

st.set_page_config(page_title="FraudGuard | Fraud Analytics", page_icon="🛡️", layout="wide")

@st.cache_resource
def load():
    return joblib.load("fraud_pipeline.joblib"), json.load(open("model_meta.json"))

try:
    pipe, meta = load()
except FileNotFoundError:
    st.error("Model not found. Run the training notebook through the Save pipeline section first, then restart this app.")
    st.stop()
except Exception as exc:
    st.error(f"Model could not be loaded: {exc}")
    st.stop()
TYPES = ["TRANSFER", "CASH_OUT"]

st.title("🛡️ FraudGuard | Mobile Money Intelligence")
st.markdown("**Transaction risk scoring · Investigation queue · Cost-aware decisions**")
st.caption("Dataset: PaySim simulation. Scores are model estimates, not verified probabilities of real-world fraud.")
st.divider()
a, b, c = st.columns(3)
a.metric("Selected model", meta.get("model", "Unknown"))
b.metric("Held-out PR-AUC", f"{meta.get('test_pr_auc', 0):.3f}")
c.metric("Training-selected threshold", f"{meta['threshold']:.2f}")
st.caption("Scores transfers and cash-outs with a machine-learning model trained on the PaySim simulation. Demo – not a production system.")

st.sidebar.header("Decision settings")
threshold = st.sidebar.slider("Fraud threshold (block if score ≥)", 0.01, 0.99, float(meta["threshold"]), 0.01,
                              help="Lower = catch more fraud but block more honest users.")
fp_cost = st.sidebar.number_input("Cost of blocking an honest transaction", min_value=0.0, value=float(meta["fp_cost"]), step=1.0)
st.sidebar.info(f"Default threshold {meta['threshold']:.2f} minimizes expected cost in training.")

tab1, tab2, tab3 = st.tabs(["🔎 Single transaction", "📁 Batch scoring & dashboard", "📊 Model overview"])

with tab1:
    c1, c2, c3 = st.columns(3)
    ttype = c1.selectbox("Transaction type", TYPES)
    amount = c1.number_input("Amount", min_value=0.0, value=181.0, step=100.0)
    step = c1.number_input("Hour of simulation (step)", min_value=1, value=1, step=1)
    oo = c2.number_input("Sender balance BEFORE", min_value=0.0, value=181.0, step=100.0)
    no = c2.number_input("Sender balance AFTER", min_value=0.0, value=0.0, step=100.0)
    od = c3.number_input("Receiver balance BEFORE", min_value=0.0, value=0.0, step=100.0)
    nd = c3.number_input("Receiver balance AFTER", min_value=0.0, value=0.0, step=100.0)
    if st.button("Check transaction", type="primary"):
        row = pd.DataFrame([{"step": step, "type": ttype, "amount": amount, "oldbalanceOrg": oo,
                             "newbalanceOrig": no, "oldbalanceDest": od, "newbalanceDest": nd}])
        try:
            score = float(pipe.predict_proba(row)[0, 1])
            st.metric("Fraud score", f"{score:.1%}")
            st.progress(min(score, 1.0))
            if score >= threshold:
                st.error("🚨 High risk – block or hold for review.")
            else:
                st.success("✅ Looks normal – let it pass.")
            if oo > 0 and no == 0:
                st.warning("Sender account is fully emptied (a common fraud pattern).")
        except Exception as e:
            st.error(f"Could not score this input: {e}")

with tab2:
    st.write("Upload a CSV with columns: " + ", ".join(RAW_COLS) + ". If it also has an `isFraud` column, we show missed fraud vs. false alarms and cost.")
    up = st.file_uploader("CSV file", type="csv")
    if up is not None:
        try:
            data = pd.read_csv(up)
            if len(data) > 200_000:
                st.warning("For this demo, upload at most 200,000 transactions per file.")
                st.stop()
            missing = [c for c in RAW_COLS if c not in data.columns]
            if missing:
                st.error(f"Missing columns: {missing}")
            else:
                data = data.copy()
                if not data["type"].isin(TYPES).all():
                    st.error("This model is trained only for TRANSFER and CASH_OUT transactions. Filter other types before uploading.")
                    st.stop()
                if data[RAW_COLS].isna().any().any():
                    st.error("Missing input values found. Please clean the CSV before scoring.")
                    st.stop()
                scores = []
                for start in range(0, len(data), 10_000):
                    scores.extend(pipe.predict_proba(data.iloc[start:start+10_000][RAW_COLS])[:, 1])
                data["fraud_score"] = scores
                data["flagged"] = (data["fraud_score"] >= threshold).astype(int)
                st.write(f"Scored {len(data):,} transactions – flagged {int(data['flagged'].sum()):,}.")
                if "isFraud" in data.columns:
                    y = data["isFraud"].astype(int); f = data["flagged"]
                    tp = int(((y == 1) & (f == 1)).sum()); fn = int(((y == 1) & (f == 0)).sum())
                    fp = int(((y == 0) & (f == 1)).sum()); tn = int(((y == 0) & (f == 0)).sum())
                    missed_amt = float(data.loc[(y == 1) & (f == 0), "amount"].sum())
                    total = missed_amt + fp * fp_cost
                    a, b, c, d = st.columns(4)
                    a.metric("Frauds caught", tp); b.metric("Frauds missed", fn); c.metric("False alarms", fp); d.metric("Total cost", f"{total:,.0f}")
                    st.caption(f"Cost = missed-fraud amount ({missed_amt:,.0f}) + {fp} false alarms × {fp_cost:,.0f}")
                st.subheader("Fraud investigation dashboard")
                k1, k2, k3 = st.columns(3)
                k1.metric("Transactions", f"{len(data):,}")
                k2.metric("Flagged for review", f"{int(data['flagged'].sum()):,}")
                k3.metric("Flag rate", f"{data['flagged'].mean():.2%}")
                c_left, c_right = st.columns(2)
                with c_left:
                    st.write("**Score distribution**")
                    st.bar_chart(pd.cut(data['fraud_score'], bins=np.linspace(0, 1, 21), include_lowest=True).value_counts().sort_index())
                with c_right:
                    st.write("**Flagged by transaction type**")
                    st.bar_chart(data.groupby('type')['flagged'].agg(['sum', 'count']))
                if 'isFraud' in data.columns and set(data['isFraud'].dropna().unique()).issubset({0,1}):
                    st.subheader("Threshold experiment: missed fraud vs false alarms")
                    st.caption("Ground truth required; cost assumptions are editable and not validated business losses.")
                    y_arr = data['isFraud'].to_numpy(dtype=int)
                    s_arr = data['fraud_score'].to_numpy()
                    amt_arr = data['amount'].to_numpy(dtype=float)
                    cost_rows = []
                    for t in np.linspace(0.01, 0.99, 99):
                        flags = s_arr >= t
                        missed = int(((y_arr == 1) & ~flags).sum())
                        false = int(((y_arr == 0) & flags).sum())
                        loss = float(amt_arr[(y_arr == 1) & ~flags].sum() + false * fp_cost)
                        cost_rows.append({'threshold': float(t), 'missed fraud': missed, 'false alarms': false, 'total cost': loss})
                    curves = pd.DataFrame(cost_rows).set_index('threshold')
                    st.line_chart(curves[['missed fraud', 'false alarms']])
                    st.line_chart(curves[['total cost']])
                    best_t = curves['total cost'].idxmin()
                    st.info(f"Lowest observed cost on this uploaded labeled file: threshold {best_t:.2f}. Do not tune on the held-out test set for a reported unbiased evaluation.")
                st.write("**Highest-risk transactions**")
                st.dataframe(data.sort_values("fraud_score", ascending=False).head(200), use_container_width=True)
                st.download_button("Download scored CSV", data.to_csv(index=False), "scored.csv")
        except Exception as e:
            st.error(f"Could not read/score the file: {e}")
with tab3:
    st.subheader("Evaluation and deployment notes")
    st.write("The notebook trains baseline, logistic regression, random forest, and gradient boosting models, then compares cross-validated PR-AUC.")
    st.write("The threshold is chosen on training out-of-fold predictions using missed fraud amount + assumed false-alarm cost.")
    st.write("**Limitations:** PaySim is simulated; transaction balances may be unavailable at authorization time; scores are not calibrated; distribution shift and customer fairness need monitoring. Never automatically block real users based on this classroom demo.")
    st.write("**Recommended action:** review high-risk cases, monitor false alarms and fraud recall, and validate costs with business stakeholders.")
