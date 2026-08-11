
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="CropWise 2.0", page_icon="🌾", layout="wide")

DATA_PATHS = [
    Path("Crop_recommendation.csv"),
    Path("data/Crop_recommendation.csv"),
    Path("data/crop_recommendation.csv"),
]

FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
ALIASES = {
    "nitrogen": "N", "n": "N",
    "phosphorus": "P", "p": "P",
    "potassium": "K", "k": "K",
    "temp": "temperature", "temperature": "temperature",
    "humidity": "humidity", "ph": "ph", "rainfall": "rainfall",
}
TARGET_CANDIDATES = ["label", "crop", "Crop", "crop_name"]

def find_dataset():
    for p in DATA_PATHS:
        if p.exists():
            return p
    return None

@st.cache_data
def load_data(path_str):
    df = pd.read_csv(path_str)
    # Normalize common column spellings
    rename = {}
    for c in df.columns:
        key = str(c).strip().lower()
        if key in ALIASES:
            rename[c] = ALIASES[key]
    df = df.rename(columns=rename)
    target = next((c for c in TARGET_CANDIDATES if c in df.columns), None)
    if target is None:
        lower_map = {str(c).strip().lower(): c for c in df.columns}
        target = next((lower_map[c.lower()] for c in TARGET_CANDIDATES if c.lower() in lower_map), None)
    if target is None:
        raise ValueError("Could not find crop target column. Expected 'label' or 'crop'.")
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}. Expected N, P, K, temperature, humidity, ph, rainfall.")
    df = df[FEATURES + [target]].dropna().copy()
    df[target] = df[target].astype(str).str.strip()
    return df, target

@st.cache_resource
def train_model(path_str):
    df, target = load_data(path_str)
    X = df[FEATURES]
    y = df[target]
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.20, random_state=42, stratify=y_enc
    )
    model = RandomForestClassifier(
        n_estimators=400, random_state=42, n_jobs=-1, class_weight="balanced"
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, pred)
    report = classification_report(
        y_test, pred, target_names=le.classes_, output_dict=True, zero_division=0
    )
    return df, target, model, le, accuracy, report

def season_from_date(dt):
    m = dt.month
    if m in [12, 1, 2]:
        return "Winter"
    if m in [3, 4, 5]:
        return "Summer"
    if m in [6, 7, 8, 9]:
        return "Monsoon"
    return "Post-monsoon"

def crop_profile(df, crop):
    return df[df["crop_clean"] == crop][FEATURES].median()

def explain_prediction(df, crop, values):
    sub = df[df["crop_clean"] == crop][FEATURES]
    med = sub.median()
    msg = []
    for f in FEATURES:
        v = values[f]
        m = med[f]
        # Relative closeness to the crop's training median
        if abs(m) < 1e-9:
            continue
        rel = abs(v - m) / max(abs(m), 1e-9)
        if rel <= 0.12:
            msg.append((f, "strong", f"{f} is close to the crop's learned typical range."))
        elif rel <= 0.30:
            msg.append((f, "medium", f"{f} is reasonably close to the crop's learned profile."))
        else:
            msg.append((f, "weak", f"{f} differs noticeably from the crop's learned profile."))
    return msg

def what_if(df, crop, values):
    sub = df[df["crop_clean"] == crop][FEATURES]
    med = sub.median()
    rows = []
    for f in FEATURES:
        current = float(values[f])
        target = float(med[f])
        if abs(current - target) < 1e-9:
            action = "Already close"
        elif current < target:
            action = f"Increase toward ~{target:.1f}"
        else:
            action = f"Decrease toward ~{target:.1f}"
        rows.append([f, round(current, 2), round(target, 2), action])
    return pd.DataFrame(rows, columns=["Factor", "Current", "Typical for crop", "Scenario guidance"])

# ---------- UI ----------
st.markdown("""
<style>
.main {background: #f7fbf5;}
.hero {padding: 22px 28px; border-radius: 18px; background: linear-gradient(135deg,#0f5132,#2e8b57); color:white;}
.card {padding:18px; border-radius:16px; background:white; border:1px solid #e5e7eb; margin-bottom:12px;}
.small {color:#667085; font-size:0.9rem;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="hero"><h1>🌾 CropWise 2.0</h1><p>Explainable Crop Classifier & Farm Decision Support System</p></div>', unsafe_allow_html=True)
st.write("")

with st.sidebar:
    st.header("🔐 CropWise Login")
    username = st.text_input("User ID", value="farmer")
    password = st.text_input("Password", type="password", value="cropwise")
    if username != "farmer" or password != "cropwise":
        st.warning("Demo login: farmer / cropwise")
        st.stop()
    st.success("Logged in")

dataset = find_dataset()
if dataset is None:
    st.error("Dataset not found.")
    st.markdown("""
    ### Add your dataset
    Put your standard crop recommendation CSV in this project as:
    `Crop_recommendation.csv`

    Required columns:
    `N, P, K, temperature, humidity, ph, rainfall, label`
    """)
    st.stop()

try:
    df, target, model, le, accuracy, report = train_model(str(dataset))
except Exception as e:
    st.error(f"Could not train the model: {e}")
    st.stop()

df["crop_clean"] = df[target].astype(str).str.strip()

tab1, tab2, tab3 = st.tabs(["🌱 Classifier", "🔄 What-if Lab", "📊 Model"])

with tab1:
    st.subheader("Farm conditions")
    c1, c2, c3 = st.columns(3)
    with c1:
        N = st.number_input("Nitrogen (N)", 0.0, 200.0, 90.0)
        P = st.number_input("Phosphorus (P)", 0.0, 200.0, 42.0)
        K = st.number_input("Potassium (K)", 0.0, 250.0, 43.0)
    with c2:
        temperature = st.number_input("Temperature (°C)", -10.0, 60.0, 25.0)
        humidity = st.number_input("Humidity (%)", 0.0, 100.0, 70.0)
        ph = st.number_input("Soil pH", 0.0, 14.0, 6.5)
    with c3:
        rainfall = st.number_input("Rainfall (mm)", 0.0, 1000.0, 200.0)
        farm_date = st.date_input("Farm assessment date")
        st.info(f"Detected season: **{season_from_date(farm_date)}**")

    values = {"N": N, "P": P, "K": K, "temperature": temperature,
              "humidity": humidity, "ph": ph, "rainfall": rainfall}
    X_input = pd.DataFrame([values])[FEATURES]

    if st.button("🌾 Analyze My Farm", type="primary", use_container_width=True):
        probs = model.predict_proba(X_input)[0]
        order = np.argsort(probs)[::-1][:3]
        results = [(le.classes_[i], float(probs[i]) * 100) for i in order]
        best_crop, best_score = results[0]

        st.success(f"🥇 Best crop: **{best_crop.title()}** — **{best_score:.1f}% model confidence**")
        cols = st.columns(3)
        medals = ["🥇", "🥈", "🥉"]
        for col, medal, (crop, score) in zip(cols, medals, results):
            with col:
                st.metric(f"{medal} {crop.title()}", f"{score:.1f}%")

        st.subheader(f"💡 Why {best_crop.title()}?")
        reasons = explain_prediction(df, best_crop, values)
        good = sum(1 for _, level, _ in reasons if level == "strong")
        for f, level, text in reasons:
            icon = "🟢" if level == "strong" else ("🟡" if level == "medium" else "🔴")
            st.write(f"{icon} {text}")

        st.subheader("📌 Farm summary")
        st.dataframe(pd.DataFrame([values]), use_container_width=True, hide_index=True)

        st.caption("Confidence is the trained classifier's probability estimate; it is not a guarantee of yield, profit, or real-world success.")

with tab2:
    st.subheader("🔄 What-if Lab")
    crop_options = sorted(df["crop_clean"].unique())
    desired = st.selectbox("Choose a crop you want to explore", crop_options)
    st.write("CropWise compares your current inputs with the median profile learned from the training data.")
    if st.button("Run What-if Analysis", use_container_width=True):
        values = {
            "N": st.session_state.get("N", 90.0),
            "P": st.session_state.get("P", 42.0),
            "K": st.session_state.get("K", 43.0),
            "temperature": st.session_state.get("temperature", 25.0),
            "humidity": st.session_state.get("humidity", 70.0),
            "ph": st.session_state.get("ph", 6.5),
            "rainfall": st.session_state.get("rainfall", 200.0),
        }
        st.dataframe(what_if(df, desired, values), use_container_width=True, hide_index=True)
        st.info("This is scenario guidance based on training-data patterns, not a fertilizer prescription or guaranteed future prediction.")

with tab3:
    st.subheader("📊 Model transparency")
    m1, m2, m3 = st.columns(3)
    m1.metric("Test accuracy", f"{accuracy*100:.2f}%")
    m2.metric("Crops/classes", len(le.classes_))
    m3.metric("Training rows", len(df))

    imp = pd.DataFrame({"Feature": FEATURES, "Importance": model.feature_importances_}).sort_values("Importance", ascending=False)
    st.bar_chart(imp.set_index("Feature"))
    st.write("Top feature influences are global Random Forest feature importances; they explain the model overall, not a single prediction.")
    st.dataframe(imp, use_container_width=True, hide_index=True)

st.divider()
st.caption("CropWise 2.0 • Educational decision-support prototype • Validate recommendations with local agronomists and soil testing before real farm decisions.")
