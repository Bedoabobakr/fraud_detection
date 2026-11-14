# Feature engineering used by BOTH the notebook and the Streamlit app (saved inside the pipeline).
import numpy as np
import pandas as pd

RAW_COLS = ["step", "type", "amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]
MONEY_COLS = ["amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest", "errorOrig", "errorDest"]
FLAG_COLS = ["origEmptied", "destZeroBefore", "destZeroAfter", "amountEqOldBal"]
NUM_COLS = MONEY_COLS + ["hour", "amountToBalance"] + FLAG_COLS

def _slog(s):
    # signed log1p: tames heavy-tailed money values, keeps sign and zero.
    return np.sign(s) * np.log1p(np.abs(s))

def add_features(X):
    X = pd.DataFrame(X).copy()
    for c in RAW_COLS:
        if c not in X.columns:
            raise ValueError(f"Missing column: {c}")
    X["hour"] = X["step"] % 24
    err_o = X["newbalanceOrig"] + X["amount"] - X["oldbalanceOrg"]       # 0 when sender books are consistent
    err_d = X["oldbalanceDest"] + X["amount"] - X["newbalanceDest"]      # 0 when receiver books are consistent
    X["origEmptied"] = ((X["oldbalanceOrg"] > 0) & (X["newbalanceOrig"] == 0)).astype(int)
    X["destZeroBefore"] = (X["oldbalanceDest"] == 0).astype(int)
    X["destZeroAfter"] = (X["newbalanceDest"] == 0).astype(int)
    X["amountEqOldBal"] = (np.isclose(X["amount"], X["oldbalanceOrg"])).astype(int)
    X["amountToBalance"] = X["amount"] / (X["oldbalanceOrg"] + 1)
    X["errorOrig"], X["errorDest"] = err_o, err_d
    for c in MONEY_COLS:
        X[c] = _slog(X[c])
    return X[["type"] + NUM_COLS]
