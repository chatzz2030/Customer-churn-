import streamlit as st
import joblib
import numpy as np
import pandas as pd

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Background */
    .stApp { background: linear-gradient(135deg, #0f0c29, #302b63, #24243e); }

    /* Main card */
    .main-card {
        background: rgba(255,255,255,0.07);
        border: 1px solid rgba(255,255,255,0.15);
        border-radius: 20px;
        padding: 2rem 2.5rem;
        backdrop-filter: blur(12px);
        margin-bottom: 1.5rem;
    }

    /* Title */
    .hero-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.3rem;
    }
    .hero-sub {
        text-align: center;
        color: #94a3b8;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }

    /* Result boxes */
    .result-churn {
        background: linear-gradient(135deg, #7f1d1d, #991b1b);
        border: 1px solid #f87171;
        border-radius: 16px;
        padding: 1.8rem;
        text-align: center;
        animation: pulse 1.5s infinite;
    }
    .result-no-churn {
        background: linear-gradient(135deg, #064e3b, #065f46);
        border: 1px solid #34d399;
        border-radius: 16px;
        padding: 1.8rem;
        text-align: center;
    }
    .result-title { font-size: 1.4rem; font-weight: 700; color: #f1f5f9; }
    .result-emoji { font-size: 3.5rem; margin: 0.5rem 0; }
    .result-prob  { font-size: 1.1rem; color: #e2e8f0; margin-top: 0.5rem; }

    @keyframes pulse {
        0%,100% { box-shadow: 0 0 0 0 rgba(248,113,113,.4); }
        50%      { box-shadow: 0 0 0 12px rgba(248,113,113,0); }
    }

    /* Section headers */
    .section-header {
        font-size: 1.1rem;
        font-weight: 700;
        color: #a78bfa;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin: 1.2rem 0 0.6rem 0;
        border-bottom: 1px solid rgba(167,139,250,0.3);
        padding-bottom: 0.3rem;
    }

    /* Metric cards */
    .metric-grid { display: flex; gap: 1rem; flex-wrap: wrap; margin-bottom: 1rem; }
    .metric-card {
        flex: 1; min-width: 120px;
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 12px;
        padding: 0.9rem 1.2rem;
        text-align: center;
    }
    .metric-label { font-size: 0.75rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.5px; }
    .metric-value { font-size: 1.5rem; font-weight: 700; color: #e2e8f0; margin-top: 0.25rem; }

    /* Sidebar styling */
    section[data-testid="stSidebar"] { background: rgba(15,12,41,0.85); }
    section[data-testid="stSidebar"] label { color: #cbd5e1 !important; }

    /* Slider & select overrides */
    .stSlider > div > div > div { background: #a78bfa !important; }

    /* Button */
    .stButton > button {
        background: linear-gradient(90deg, #7c3aed, #2563eb);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-size: 1.05rem;
        font-weight: 600;
        width: 100%;
        transition: opacity 0.2s;
    }
    .stButton > button:hover { opacity: 0.88; }

    /* Divider */
    hr { border-color: rgba(255,255,255,0.1); }

    /* Feature importance bars */
    .bar-row { display: flex; align-items: center; margin: 0.35rem 0; }
    .bar-label { color: #cbd5e1; font-size: 0.85rem; width: 220px; flex-shrink: 0; }
    .bar-track { flex: 1; background: rgba(255,255,255,0.08); border-radius: 99px; height: 10px; overflow: hidden; }
    .bar-fill  { height: 100%; border-radius: 99px; background: linear-gradient(90deg, #7c3aed, #60a5fa); }
</style>
""", unsafe_allow_html=True)

# ─── Load Model ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

model = load_model()

# ─── Hero ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">📊 Customer Churn Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Logistic Regression · IBM Telco Dataset · 80% Accuracy</div>', unsafe_allow_html=True)
st.markdown("---")

# ─── Sidebar Inputs ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧾 Customer Profile")
    st.markdown("Fill in the customer details below to get a churn prediction.")
    st.markdown("---")

    # ── Demographics ──────────────────────────────────────────────────────────
    st.markdown("### 👤 Demographics")
    gender = st.selectbox("Gender", ["Female", "Male"])
    senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner = st.selectbox("Has Partner?", ["No", "Yes"])
    dependents = st.selectbox("Has Dependents?", ["No", "Yes"])

    # ── Account Info ──────────────────────────────────────────────────────────
    st.markdown("### 🗂️ Account Info")
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.selectbox("Paperless Billing?", ["No", "Yes"])
    payment_method = st.selectbox("Payment Method", [
        "Bank transfer (automatic)",
        "Credit card (automatic)",
        "Electronic check",
        "Mailed check"
    ])

    # ── Services ──────────────────────────────────────────────────────────────
    st.markdown("### 📡 Services")
    phone_service = st.selectbox("Phone Service?", ["No", "Yes"])
    multiple_lines = st.selectbox("Multiple Lines?", ["No", "Yes", "No phone service"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox("Online Security?", ["No", "Yes", "No internet service"])
    online_backup = st.selectbox("Online Backup?", ["No", "Yes", "No internet service"])
    device_protection = st.selectbox("Device Protection?", ["No", "Yes", "No internet service"])
    tech_support = st.selectbox("Tech Support?", ["No", "Yes", "No internet service"])
    streaming_tv = st.selectbox("Streaming TV?", ["No", "Yes", "No internet service"])
    streaming_movies = st.selectbox("Streaming Movies?", ["No", "Yes", "No internet service"])

    # ── Billing ───────────────────────────────────────────────────────────────
    st.markdown("### 💳 Billing")
    monthly_charges = st.slider("Monthly Charges ($)", 18.0, 120.0, 65.0, step=0.5)
    total_charges = st.slider("Total Charges ($)", 0.0, 8700.0, float(tenure * monthly_charges), step=10.0)

    predict_btn = st.button("🔮 Predict Churn")

# ─── Feature Engineering (mirror the notebook exactly) ───────────────────────
def build_feature_vector(
    gender, senior_citizen, partner, dependents, tenure,
    phone_service, multiple_lines, internet_service,
    online_security, online_backup, device_protection, tech_support,
    streaming_tv, streaming_movies, contract, paperless_billing,
    payment_method, monthly_charges, total_charges
):
    """
    Reproduce the exact pd.get_dummies(drop_first=True) encoding used in the notebook.
    Column order (30 features, verified from notebook output):
      SeniorCitizen, tenure, MonthlyCharges, TotalCharges,
      gender_Male, Partner_Yes, Dependents_Yes, PhoneService_Yes,
      MultipleLines_No phone service, MultipleLines_Yes,
      InternetService_Fiber optic, InternetService_No,
      OnlineSecurity_No internet service, OnlineSecurity_Yes,
      OnlineBackup_No internet service, OnlineBackup_Yes,
      DeviceProtection_No internet service, DeviceProtection_Yes,
      TechSupport_No internet service, TechSupport_Yes,
      StreamingTV_No internet service, StreamingTV_Yes,
      StreamingMovies_No internet service, StreamingMovies_Yes,
      Contract_One year, Contract_Two year,
      PaperlessBilling_Yes,
      PaymentMethod_Credit card (automatic),
      PaymentMethod_Electronic check,
      PaymentMethod_Mailed check
    Numerical cols (StandardScaler): SeniorCitizen, tenure, MonthlyCharges, TotalCharges
    """
    # Raw numerical
    sc = 1 if senior_citizen == "Yes" else 0
    t  = tenure
    mc = monthly_charges
    tc = total_charges

    # Scaling params from training data (IBM Telco full set, 7032 rows)
    # Order: SeniorCitizen, tenure, MonthlyCharges, TotalCharges
    MEANS = [0.1624,    32.421786,  64.798208,   2283.300441]
    STDS  = [0.368844,  24.54526,   30.085974,   2266.771362]

    sc_scaled = (sc - MEANS[0]) / STDS[0]
    t_scaled  = (t  - MEANS[1]) / STDS[1]
    mc_scaled = (mc - MEANS[2]) / STDS[2]
    tc_scaled = (tc - MEANS[3]) / STDS[3]

    def yn(val):  return 1 if val == "Yes" else 0
    def eq(val, target): return 1 if val == target else 0

    row = [
        sc_scaled, t_scaled, mc_scaled, tc_scaled,
        eq(gender,           "Male"),
        yn(partner),
        yn(dependents),
        yn(phone_service),
        eq(multiple_lines,   "No phone service"),
        eq(multiple_lines,   "Yes"),
        eq(internet_service, "Fiber optic"),
        eq(internet_service, "No"),
        eq(online_security,  "No internet service"),
        eq(online_security,  "Yes"),
        eq(online_backup,    "No internet service"),
        eq(online_backup,    "Yes"),
        eq(device_protection,"No internet service"),
        eq(device_protection,"Yes"),
        eq(tech_support,     "No internet service"),
        eq(tech_support,     "Yes"),
        eq(streaming_tv,     "No internet service"),
        eq(streaming_tv,     "Yes"),
        eq(streaming_movies, "No internet service"),
        eq(streaming_movies, "Yes"),
        eq(contract,         "One year"),
        eq(contract,         "Two year"),
        yn(paperless_billing),
        eq(payment_method,   "Credit card (automatic)"),
        eq(payment_method,   "Electronic check"),
        eq(payment_method,   "Mailed check"),
    ]
    return np.array(row, dtype=float).reshape(1, -1)

# ─── Main Content ─────────────────────────────────────────────────────────────
col_left, col_right = st.columns([3, 2], gap="large")

with col_left:
    # ── Model Info ────────────────────────────────────────────────────────────
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">📈 Model Performance</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="metric-grid">
        <div class="metric-card">
            <div class="metric-label">Accuracy</div>
            <div class="metric-value">80.4%</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Precision (Churn)</div>
            <div class="metric-value">65%</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Recall (Churn)</div>
            <div class="metric-value">57%</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">F1-Score</div>
            <div class="metric-value">0.61</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Dataset</div>
            <div class="metric-value">7,032</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Key Risk Factors ──────────────────────────────────────────────────────
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">⚠️ Top Churn Risk Factors</div>', unsafe_allow_html=True)

    factors = [
        ("Month-to-Month Contract", 92),
        ("Fiber Optic Internet",    78),
        ("Electronic Check Payment",72),
        ("Short Tenure (< 6 months)",68),
        ("No Tech Support",          61),
        ("No Online Security",       58),
        ("Paperless Billing",        45),
        ("High Monthly Charges",     43),
    ]
    for label, pct in factors:
        st.markdown(f"""
        <div class="bar-row">
            <div class="bar-label">{label}</div>
            <div class="bar-track">
                <div class="bar-fill" style="width:{pct}%"></div>
            </div>
            <span style="color:#94a3b8;font-size:0.8rem;margin-left:8px">{pct}%</span>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # ── About the Project ─────────────────────────────────────────────────────
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">🔬 About This Project</div>', unsafe_allow_html=True)
    st.markdown("""
    <ul style="color:#cbd5e1; line-height:1.9; padding-left:1.2rem">
        <li><b>Dataset:</b> IBM Telco Customer Churn (7,032 customers, 21 features)</li>
        <li><b>EDA:</b> Pie chart of churn rate (~26%), contract type vs churn bar chart, tenure histogram, monthly charges KDE</li>
        <li><b>Pre-processing:</b> Dropped customerID, converted TotalCharges to numeric, dropped NaNs, one-hot encoded categoricals, StandardScaler on numericals</li>
        <li><b>Models trained:</b> Logistic Regression (80.4% acc) &amp; Random Forest (79.0% acc)</li>
        <li><b>Saved model:</b> <code>model.pkl</code> (Logistic Regression via joblib)</li>
    </ul>
    """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col_right:
    # ── Prediction Result ─────────────────────────────────────────────────────
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">🔮 Prediction</div>', unsafe_allow_html=True)

    if predict_btn:
        X = build_feature_vector(
            gender, senior_citizen, partner, dependents, tenure,
            phone_service, multiple_lines, internet_service,
            online_security, online_backup, device_protection, tech_support,
            streaming_tv, streaming_movies, contract, paperless_billing,
            payment_method, monthly_charges, total_charges
        )
        pred = model.predict(X)[0]
        prob = model.predict_proba(X)[0]
        churn_prob   = round(prob[1] * 100, 1)
        no_churn_prob = round(prob[0] * 100, 1)

        if pred == 1:
            st.markdown(f"""
            <div class="result-churn">
                <div class="result-emoji">🚨</div>
                <div class="result-title">HIGH CHURN RISK</div>
                <div class="result-prob">Churn Probability: <b>{churn_prob}%</b></div>
                <div class="result-prob" style="font-size:0.9rem;margin-top:0.4rem;color:#fca5a5">
                    This customer is likely to leave soon.
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-no-churn">
                <div class="result-emoji">✅</div>
                <div class="result-title">LOW CHURN RISK</div>
                <div class="result-prob">Retention Probability: <b>{no_churn_prob}%</b></div>
                <div class="result-prob" style="font-size:0.9rem;margin-top:0.4rem;color:#6ee7b7">
                    This customer is likely to stay.
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Probability gauge
        st.markdown("**Churn Probability Breakdown**")
        st.progress(int(churn_prob))
        c1, c2 = st.columns(2)
        c1.metric("🔴 Churn",    f"{churn_prob}%")
        c2.metric("🟢 No Churn", f"{no_churn_prob}%")

        # Input summary
        st.markdown("---")
        st.markdown("**Input Summary**")
        summary = {
            "Tenure": f"{tenure} months",
            "Contract": contract,
            "Internet": internet_service,
            "Monthly $": f"${monthly_charges:.2f}",
            "Payment": payment_method,
        }
        for k, v in summary.items():
            st.markdown(f"- **{k}:** {v}")

    else:
        st.markdown("""
        <div style="text-align:center; padding: 3rem 1rem; color:#64748b;">
            <div style="font-size:3rem;">👈</div>
            <div style="font-size:1rem; margin-top:0.5rem;">
                Fill in the customer details in the<br>sidebar and click <b>Predict Churn</b>.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # ── Quick Tips ────────────────────────────────────────────────────────────
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">💡 Retention Tips</div>', unsafe_allow_html=True)
    st.markdown("""
    <ul style="color:#cbd5e1; line-height:1.9; padding-left:1.2rem; font-size:0.9rem">
        <li>Offer <b>annual contracts</b> with a discount</li>
        <li>Bundle <b>Tech Support + Online Security</b></li>
        <li>Proactively reach out at <b>month 3-6</b></li>
        <li>Switch customers from Electronic Check to <b>Auto-Pay</b></li>
        <li>Targeted offers for <b>Fiber Optic</b> users</li>
    </ul>
    """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


