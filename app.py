
import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib

# Try to import sklearn, otherwise instruct user to install requirements
try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
except Exception as e:
    LogisticRegression = None
    train_test_split = None
    accuracy_score = None
    roc_auc_score = None
    confusion_matrix = None

# File paths
HISTORY_FILE = "history.csv"
RESULTS_FILE = "results.csv"
MODEL_FILE = "ml_model.pkl"
FLAT_FILE = "sol_balance.txt"
FIXED_FILE = "fixed_balance.txt"

# Constants
INITIAL_BALANCE = 0.1
FLAT_BET = 0.01
FIXED_BET = 0.02
WINDOW = 20  # use last 20 rounds for edge detector / features
EXPECTED_UNDER_RATE = 0.65

# --- Data helpers ---
@st.cache_data
def load_csv(file):
    df = pd.read_csv(file)
    return df['multiplier'].tolist()

def load_history():
    if os.path.exists(HISTORY_FILE):
        df = pd.read_csv(HISTORY_FILE)
        return df['multiplier'].tolist()
    return []

def save_history(data):
    pd.DataFrame({'multiplier': data}).to_csv(HISTORY_FILE, index=False)

def load_results():
    if os.path.exists(RESULTS_FILE):
        return pd.read_csv(RESULTS_FILE)
    return pd.DataFrame(columns=['prediction', 'actual', 'correct'])

def save_result(prediction, actual):
    correct = ((prediction == "Above") and actual > 2.0) or ((prediction == "Under") and actual <= 2.0)
    result_df = load_results()
    result_df.loc[len(result_df)] = [prediction, actual, correct]
    result_df.to_csv(RESULTS_FILE, index=False)
    update_flat_balance(prediction, actual)
    if prediction == "Above":
        update_fixed_balance(actual)

# --- Balance handlers ---
def get_flat_balance():
    if os.path.exists(FLAT_FILE):
        with open(FLAT_FILE, "r") as f:
            return float(f.read())
    return INITIAL_BALANCE

def get_fixed_balance():
    if os.path.exists(FIXED_FILE):
        with open(FIXED_FILE, "r") as f:
            return float(f.read())
    return INITIAL_BALANCE

def update_flat_balance(prediction, actual):
    balance = get_flat_balance()
    if prediction == "Above":
        balance += FLAT_BET if actual > 2.0 else -FLAT_BET
        with open(FLAT_FILE, "w") as f:
            f.write(str(balance))

def update_fixed_balance(actual):
    balance = get_fixed_balance()
    balance += FIXED_BET if actual > 2.0 else -FIXED_BET
    with open(FIXED_FILE, "w") as f:
        f.write(str(balance))

def reset_balance():
    for f in [FLAT_FILE, FIXED_FILE, MODEL_FILE, RESULTS_FILE]:
        if os.path.exists(f):
            os.remove(f)

# --- Feature engineering ---
def create_features(series, window=WINDOW):
    """
    Given a list/array of multipliers, create tabular features and labels for ML:
    For each index i where i+1 exists, build features from previous `window` values
    and label whether next multiplier > 2.0
    """
    X = []
    y = []
    n = len(series)
    for i in range(window, n-1):
        window_vals = np.array(series[i-window:i])
        features = {
            'mean': window_vals.mean(),
            'median': np.median(window_vals),
            'std': window_vals.std(),
            'min': window_vals.min(),
            'max': window_vals.max(),
            'under_count': np.sum(window_vals < 2.0),
            'above_count': np.sum(window_vals >= 2.0),
            'last': series[i-1],
            'last2': series[i-2] if i-2 >=0 else series[i-1],
            'last3': series[i-3] if i-3 >=0 else series[i-1],
            'momentum': window_vals[-1] - window_vals[0],
        }
        X.append(list(features.values()))
        y.append(1 if series[i+1] > 2.0 else 0)
    cols = list(features.keys())
    return np.array(X), np.array(y), cols

# --- Edge detector using last WINDOW rounds ---
def detect_statistical_edge(data, threshold=2.0, window=WINDOW, expected_under_rate=EXPECTED_UNDER_RATE):
    if len(data) < window:
        return {
            "rolling_avg": None,
            "under_count": None,
            "z_score": None,
            "edge_detected": False,
        }
    recent = np.array(data[-window:])
    under_count = np.sum(recent < threshold)
    rolling_avg = np.mean(recent)
    expected_mean = expected_under_rate * window
    std_dev = np.sqrt(window * expected_under_rate * (1 - expected_under_rate))
    z_score = (under_count - expected_mean) / std_dev if std_dev > 0 else 0
    # Use more conservative thresholds for WINDOW=20
    edge_detected = (rolling_avg < 1.6) and (under_count >= int(0.7*window)) and (z_score >= 1.5)
    return {
        "rolling_avg": round(rolling_avg, 3),
        "under_count": int(under_count),
        "z_score": round(z_score, 3),
        "edge_detected": edge_detected,
    }

# --- ML model helpers ---
def train_model_from_history(series):
    if LogisticRegression is None:
        return None, "scikit-learn not installed"
    X, y, cols = create_features(series)
    if len(y) < 10:
        return None, "Not enough samples to train (need at least 10)."
    # handle case where y has single class
    if len(np.unique(y)) == 1:
        return None, "Labels are all one class; cannot train."
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:,1]
    acc = accuracy_score(y_test, preds) if accuracy_score else None
    auc = roc_auc_score(y_test, probs) if roc_auc_score else None
    # save model and columns
    joblib.dump({'model': model, 'cols': cols}, MODEL_FILE)
    return {'model': model, 'cols': cols, 'acc': acc, 'auc': auc}, None

def load_model():
    if os.path.exists(MODEL_FILE):
        d = joblib.load(MODEL_FILE)
        return d['model'], d['cols']
    return None, None

def predict_next_with_model(series):
    model, cols = load_model()
    if model is None:
        return None, None
    if len(series) < WINDOW:
        return None, None
    # Prepare feature vector from last WINDOW values
    window_vals = np.array(series[-WINDOW:])
    features = [
        window_vals.mean(),
        np.median(window_vals),
        window_vals.std(),
        window_vals.min(),
        window_vals.max(),
        int(np.sum(window_vals < 2.0)),
        int(np.sum(window_vals >= 2.0)),
        series[-1],
        series[-2] if len(series)>=2 else series[-1],
        series[-3] if len(series)>=3 else series[-1],
        window_vals[-1] - window_vals[0],
    ]
    prob = model.predict_proba([features])[0][1]
    pred = 1 if prob > 0.5 else 0
    return pred, prob

# --- Streamlit UI ---
def main():
    st.title("Crash Edge Detector — Last 20 + ML")

    if "history" not in st.session_state:
        st.session_state.history = load_history()

    col1, col2 = st.columns([2,1])
    with col1:
        uploaded = st.file_uploader("Upload multipliers CSV (one column 'multiplier')", type=["csv"])
        if uploaded:
            st.session_state.history = load_csv(uploaded)
            save_history(st.session_state.history)
            st.success(f"Loaded {len(st.session_state.history)} multipliers.")

    with col2:
        if st.button("Reset all (balances, model, results)"):
            st.session_state.history = []
            save_history([])
            reset_balance()
            st.success("Reset done.")

    st.subheader("Manual input")
    new_val = st.text_input("Enter multiplier (e.g., 1.87 or 187)")
    if st.button("Add multiplier"):
        try:
            val = float(new_val)
            val = val/100 if val>10 else val
            if "last_prediction" in st.session_state:
                save_result(st.session_state.last_prediction, val)
                del st.session_state.last_prediction
            st.session_state.history.append(val)
            save_history(st.session_state.history)
            st.success(f"Added {val}x")
        except Exception as e:
            st.error("Invalid input")

    if st.session_state.history:
        data = st.session_state.history
        st.write(f"History length: {len(data)}")

        # Show edge detector (last WINDOW)
        st.subheader("Statistical Edge Detector (last 20)")
        edge = detect_statistical_edge(data)
        if edge["rolling_avg"] is not None:
            st.write(f"Rolling average (last {WINDOW}): **{edge['rolling_avg']}x**")
            st.write(f"Under-2.0x count: **{edge['under_count']}** / {WINDOW}")
            st.write(f"Z-score: **{edge['z_score']}**")
            if edge['edge_detected']:
                st.success("✅ Edge detected (conservative thresholds). Consider betting 'Above'")
            else:
                st.info("No clear edge detected.")

        # Train model section
        st.subheader("Machine Learning Model")
        st.write("Create/train a logistic regression model to predict whether next multiplier > 2.0x.")
        model, cols = load_model()
        if model is None:
            st.info("No trained model found. Train one from history below.")
        else:
            st.success("Trained model loaded.")
            st.write(f"Feature columns used: {cols}")

        if st.button("Train model from history"):
            with st.spinner("Training..."):
                res, err = train_model_from_history(data)
                if err:
                    st.error(err)
                else:
                    st.success(f"Model trained. Acc (test): {res['acc']:.3f} AUC: {res['auc']:.3f if res['auc'] is not None else 'N/A'}")

        # Predict next using model
        pred, prob = predict_next_with_model(data)
        if prob is not None:
            st.write(f"ML predicted probability next >2.0x: **{prob:.2%}**")
            st.write(f"ML prediction: **{'Above' if pred==1 else 'Under'}**")
            # Display an action recommendation with conservative threshold
            if prob > 0.7 and edge['edge_detected']:
                st.success("Strong signal: ML probability high AND statistical edge present. Consider betting.")
            elif prob > 0.75:
                st.warning("ML probability high but no statistical edge detected. Be cautious.")
            elif edge['edge_detected']:
                st.info("Statistical edge present but ML probability not high. Consider small bet or wait.")
            else:
                st.write("No strong combined signal. Recommend skipping bet.")

        # Accuracy tracker and results
        st.subheader("Accuracy & balances")
        results_df = load_results()
        if not results_df.empty:
            total = len(results_df)
            correct = results_df['correct'].sum()
            acc = correct / total if total else 0
            st.metric("Total Predictions", total)
            st.metric("Correct Predictions", int(correct))
            st.metric("Accuracy Rate", f"{acc:.1%}")
            st.dataframe(results_df[::-1].reset_index(drop=True))
            # breakdown for 'Above' preds
            above_preds = results_df[results_df["prediction"]=="Above"]
            if not above_preds.empty:
                st.metric("Above Pred Count", len(above_preds))
                st.metric("Above Win Rate", f"{above_preds['correct'].mean():.1%}")
        else:
            st.write("No verified predictions yet.")

        # Balances
        st.subheader("Balances")
        st.metric("Flat Bet Balance (0.01 SOL per 'Above')", f"{get_flat_balance():.4f} SOL")
        st.metric("Fixed Bet Balance (0.02 SOL only when 'Above')", f"{get_fixed_balance():.4f} SOL")
        st.caption("Bets applied only when you record a prediction and then add the actual result.")

    else:
        st.write("No history yet. Upload CSV or add multipliers manually.")

if __name__ == "__main__":
    main()
