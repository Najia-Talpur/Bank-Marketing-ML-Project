import streamlit as st
import pandas as pd
import joblib


st.set_page_config(
    page_title="Bank Marketing Prediction",
    page_icon="🏦",
    layout="wide"
)


st.markdown("""
<style>
    .stApp {
        background-color: #F5F7FB;
        color: #172B4D;
    }

    .block-container {
        padding-top: 2rem;
        max-width: 1100px;
    }

    /* Main header */
    .hero {
        background: linear-gradient(120deg, #102A43, #176B87);
        padding: 28px;
        border-radius: 18px;
        margin-bottom: 25px;
    }

    .hero h1 {
        color: #FFFFFF !important;
        font-size: 34px;
        margin-bottom: 8px;
    }

    .hero p {
        color: #E6F6FF !important;
        font-size: 16px;
        margin-bottom: 0;
    }

    /* Section headings */
    h2, h3 {
        color: #102A43 !important;
    }

    /* Normal text and labels */
    p, label, .stMarkdown, 
    div[data-testid="stCaptionContainer"] {
        color: #243B53;
    }

    /* Small explanatory text */
    .help-text {
        color: #526477 !important;
        font-size: 12px;
        margin-top: -8px;
        margin-bottom: 12px;
    }

    /* Input fields */
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div {
        background-color: #FFFFFF;
        border-radius: 8px;
    }

    /* Button */
    div.stButton > button {
        background-color: #176B87;
        color: #FFFFFF;
        border: none;
        border-radius: 9px;
        height: 46px;
        font-weight: 600;
    }

    div.stButton > button:hover {
        background-color: #102A43;
        color: #FFFFFF;
    }

    /* Result cards */
    .result-note {
        background-color: #E8EEF5;
        padding: 14px;
        border-radius: 10px;
        color: #243B53 !important;
        font-size: 14px;
    }
</style>
""", unsafe_allow_html=True)



model = joblib.load("RF_model_bank_marketing.pkl")
scaler = joblib.load("scaler.pkl")
expected_columns = joblib.load("columns.pkl")



st.markdown("""
<div class="hero">
    <h1>🏦 Bank Marketing AI</h1>
    <p>
        Predict whether a customer may subscribe to a bank term deposit.
        Enter customer details and let the machine learning model make a prediction.
    </p>
</div>
""", unsafe_allow_html=True)


# ---------------- INTRODUCTION ----------------
st.subheader("👤 Customer Information")

st.markdown("""
<div class="result-note">
    <b>What is this app?</b><br>
    This app uses a trained Random Forest classification model to estimate
    whether a customer will subscribe to a term deposit after a bank marketing
    campaign. The prediction is an estimate, not a guarantee.
</div>
""", unsafe_allow_html=True)

st.write("")


# ---------------- INPUT FIELDS ----------------
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### Personal Details")

    age = st.number_input("Age", 18, 100, 35)

    job = st.selectbox(
        "Job",
        [
            "admin.", "blue-collar", "entrepreneur",
            "housemaid", "management", "retired",
            "self-employed", "services", "student",
            "technician", "unemployed", "unknown"
        ]
    )

    marital = st.selectbox(
        "Marital Status",
        ["married", "single", "divorced"]
    )

    education = st.selectbox(
        "Education",
        ["primary", "secondary", "tertiary", "unknown"]
    )

    balance = st.number_input(
        "Account Balance",
        min_value=-100000,
        max_value=1000000,
        value=1000
    )


with col2:
    st.markdown("### Loans & Contact")

    default = st.selectbox(
        "Credit in Default?",
        ["no", "yes", "unknown"]
    )

    housing = st.selectbox(
        "Housing Loan?",
        ["no", "yes", "unknown"]
    )

    loan = st.selectbox(
        "Personal Loan?",
        ["no", "yes", "unknown"]
    )

    contact = st.selectbox(
        "Contact Type",
        ["cellular", "telephone", "unknown"]
    )

    month = st.selectbox(
        "Contact Month",
        [
            "jan", "feb", "mar", "apr", "may", "jun",
            "jul", "aug", "sep", "oct", "nov", "dec"
        ]
    )


with col3:
    st.markdown("### Campaign Details")

    day = st.number_input(
        "Contact Day",
        min_value=1,
        max_value=31,
        value=15
    )

    campaign = st.number_input(
        "Contacts in Current Campaign",
        min_value=1,
        max_value=100,
        value=1
    )

    previous = st.number_input(
        "Previous Contacts",
        min_value=0,
        max_value=100,
        value=0
    )

    pdays = st.number_input(
        "Days Since Previous Contact",
        min_value=-1,
        max_value=999,
        value=-1
    )

    poutcome = st.selectbox(
        "Previous Campaign Outcome",
        ["unknown", "failure", "other", "success"]
    )


st.markdown("---")

st.markdown("### 🔍 Make a Prediction")
st.markdown(
    '<p class="help-text">Click below to estimate whether this customer may subscribe.</p>',
    unsafe_allow_html=True
)

if st.button("Predict Subscription", type="primary"):

    input_data = pd.DataFrame([{
        "age": age,
        "job": job,
        "marital": marital,
        "education": education,
        "default": default,
        "balance": balance,
        "housing": housing,
        "loan": loan,
        "contact": contact,
        "day": day,
        "month": month,
        "campaign": campaign,
        "pdays": pdays,
        "previous": previous,
        "poutcome": poutcome
    }])

    # Encode categorical columns
    input_data = pd.get_dummies(input_data)

    # Match the feature columns used during training
    input_data = input_data.reindex(
        columns=expected_columns,
        fill_value=0
    )

    # Scale using the saved scaler
    input_scaled = scaler.transform(input_data)

    # Predict the class
    prediction = model.predict(input_scaled)[0]

    st.markdown("---")
    st.subheader("📊 Prediction Result")

    if prediction == 1:
        st.success(
            "YES — The model predicts that this customer may subscribe "
            "to a term deposit."
        )
    else:
        st.warning(
            "NO — The model predicts that this customer may not subscribe "
            "to a term deposit."
        )

    # Display probability when available
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(input_scaled)[0]
        classes = list(model.classes_)
        yes_probability = probabilities[classes.index(1)]

        st.metric(
            "Estimated Subscription Probability",
            f"{yes_probability:.1%}"
        )

        st.progress(float(yes_probability))

        st.caption(
            "This percentage is the model's estimated probability, "
            "not a guarantee of the customer's decision."
        )


st.markdown("---")
st.caption(
    "Bank Marketing AI | Random Forest Classification | Educational Project"
)




# To install streamlit:
# py -m pip install streamlit pandas joblib scikit-learn
# To rum the code
# Write on terminal:
# py -m streamlit run app.py