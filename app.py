import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pickle
from pathlib import Path

DATA_PATH = "loantrain.csv"  # ensure this file is in the same folder

@st.cache_data
def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    return df

@st.cache_data
def preprocess_dataframe(df: pd.DataFrame):
    # replicate preprocessing steps similar to notebook (fillna, log transforms)
    df = df.copy()
    # fill categorical missing with mode
    for col in ['Gender','Married','Dependents','Self_Employed']:
        if col in df.columns:
            df[col].fillna(df[col].mode()[0], inplace=True)
    # LoanAmount
    if 'LoanAmount' in df.columns:
        df['LoanAmount'] = df['LoanAmount'].fillna(df['LoanAmount'].mean())
        # add log
        df['LoanAmount_log'] = np.log(df['LoanAmount'] + 1)
    # Loan_Amount_Term and Credit_History
    if 'Loan_Amount_Term' in df.columns:
        df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0], inplace=True)
    if 'Credit_History' in df.columns:
        df['Credit_History'].fillna(df['Credit_History'].mode()[0], inplace=True)
    # Total Income features
    if set(['ApplicantIncome','CoapplicantIncome']).issubset(df.columns):
        df['TotalIncome'] = df['ApplicantIncome'] + df['CoapplicantIncome']
        df['TotalIncome_log'] = np.log(df['TotalIncome'] + 1)
    return df

@st.cache_resource
def train_models(df: pd.DataFrame):
    # We'll use a robust, commonly-used feature set:
    features = [
        'Gender','Married','Dependents','Education','Self_Employed',
        'ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term',
        'Credit_History','Property_Area'
    ]
    # keep only available features
    features = [f for f in features if f in df.columns]
    # target
    target = 'Loan_Status'
    df = df.copy()
    df = df.dropna(subset=[target])
    X = df[features].copy()
    y = df[target].copy()

    # encode categorical columns with LabelEncoder (fit on training df)
    encoders = {}
    for col in X.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le

    # encode target if needed
    target_encoder = None
    if y.dtype == 'object':
        target_encoder = LabelEncoder()
        y = target_encoder.fit_transform(y.astype(str))

    # scale numeric features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # train Decision Tree
    dt = DecisionTreeClassifier(criterion='entropy', random_state=0)
    dt.fit(X_train, y_train)
    y_pred_dt = dt.predict(X_test)
    acc_dt = accuracy_score(y_test, y_pred_dt)

    # train GaussianNB
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    y_pred_nb = nb.predict(X_test)
    acc_nb = accuracy_score(y_test, y_pred_nb)

    # store metadata for prediction step
    model_bundle = {
        'features': features,
        'encoders': encoders,
        'scaler': scaler,
        'target_encoder': target_encoder,
        'dt_model': dt,
        'nb_model': nb,
        'acc_dt': acc_dt,
        'acc_nb': acc_nb
    }
    return model_bundle

def preprocess_input(user_inputs: dict, bundle: dict):
    """
    Convert user_inputs (dict feature -> raw value) into model-ready numpy vector.
    """
    feat_order = bundle['features']
    encoders = bundle['encoders']
    scaler = bundle['scaler']
    row = []
    for f in feat_order:
        val = user_inputs.get(f)
        if f in encoders:
            le = encoders[f]
            # avoid unseen labels error: if unseen, try to map to closest or add unknown handling
            try:
                val_enc = le.transform([str(val)])[0]
            except Exception:
                # unseen label -> add by mapping to mode index 0
                val_enc = 0
            row.append(val_enc)
        else:
            # numeric
            # ensure float
            try:
                row.append(float(val))
            except:
                row.append(0.0)
    arr = np.array(row).reshape(1, -1)
    arr_scaled = scaler.transform(arr)
    return arr_scaled

# ---- Streamlit UI ----
st.set_page_config(page_title="Loan Eligibility Predictor", layout="centered")
st.title("Loan Eligibility Predictor")
st.markdown("Deploy of the Loan Eligibility project (preprocessing + models) — adapted from your notebook. :contentReference[oaicite:1]{index=1}")

# Load and prepare
if not Path(DATA_PATH).exists():
    st.error(f"Dataset not found: {DATA_PATH}. Put your `loantrain.csv` file in this folder.")
    st.stop()

raw_df = load_data(DATA_PATH)
df = preprocess_dataframe(raw_df)

st.sidebar.header("Model training & info")
if st.sidebar.button("Train / Retrain models (may take a few seconds)"):
    with st.spinner("Training models..."):
        bundle = train_models(df)
    st.success("Training completed. Models cached.")
    st.sidebar.write(f"Decision Tree accuracy (test): {bundle['acc_dt']:.3f}")
    st.sidebar.write(f"GaussianNB accuracy (test): {bundle['acc_nb']:.3f}")
else:
    # fetch trained models from cache (train_models is cached_resource)
    bundle = train_models(df)
    st.sidebar.write(f"Decision Tree accuracy (test): {bundle['acc_dt']:.3f}")
    st.sidebar.write(f"GaussianNB accuracy (test): {bundle['acc_nb']:.3f}")

# Let user pick model
model_choice = st.sidebar.selectbox("Choose model for prediction", ['GaussianNB', 'DecisionTree'])
st.sidebar.markdown("Tip: your notebook showed Naive Bayes had higher accuracy.")

st.header("Enter applicant details")

# Build input widgets for features used
inputs = {}
for f in bundle['features']:
    # choose widget type based on column dtype in original df
    if f in raw_df.columns and raw_df[f].dtype == 'object':
        opts = sorted(raw_df[f].dropna().unique().astype(str))
        inputs[f] = st.selectbox(f, options=opts, index=0, key=f"_sel_{f}")
    else:
        # numeric
        col_min = float(raw_df[f].min()) if f in raw_df.columns else 0.0
        col_max = float(raw_df[f].max()) if f in raw_df.columns else col_min + 10000.0
        col_median = float(raw_df[f].median()) if f in raw_df.columns else (col_min + col_max) / 2.0
        inputs[f] = st.number_input(f, value=col_median, min_value=0.0, step=1.0, format="%.2f", key=f"_num_{f}")

if st.button("Predict Loan Eligibility"):
    try:
        X_user = preprocess_input(inputs, bundle)
        if model_choice == 'GaussianNB':
            pred = bundle['nb_model'].predict(X_user)
        else:
            pred = bundle['dt_model'].predict(X_user)
        # decode label
        if bundle['target_encoder'] is not None:
            result = bundle['target_encoder'].inverse_transform(pred)[0]
        else:
            # numeric 1/0 fallback
            result = 'Y' if int(pred[0]) == 1 else 'N'
        st.subheader("Prediction result")
        if result in ['Y', 'y', 'Yes', 'yes', '1']:
            st.success(f"✅ Loan likely to be **Approved** (model: {model_choice})")
        else:
            st.error(f"❌ Loan likely to be **Not Approved** (model: {model_choice})")
    except Exception as e:
        st.exception(f"Prediction failed: {e}")

st.markdown("---")
st.write("You can also inspect the training dataset (first 10 rows):")
st.dataframe(df.head(10))

st.markdown("**Notes / next steps**")
st.markdown("""
- This app trains models on `loantrain.csv` on first run (cached).  
- If you want to use a pre-trained `.pkl` model instead, we can add load/save code; tell me and I'll add it.  
- The preprocessing mirrors the notebook's approach (mode fill, mean fill, log features possible). If you want the exact np.r_ feature slice used in your notebook, I can adapt the code to that exact index selection.
""")
