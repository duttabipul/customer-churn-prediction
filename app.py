from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title='Customer Churn Explorer', page_icon='📊')
st.title('Customer Churn Explorer')
st.write('Explore how a customer profile changes a model’s churn estimate.')
st.caption('Educational demo using fictional telecom data. Estimates are not calibrated or validated for real business decisions.')
if not (ROOT / 'model.joblib').exists():
    st.info('First run: python train.py')
    st.stop()

@st.cache_resource
def load_model():
    return joblib.load(ROOT / 'model.joblib')

with st.form('customer'):
    left, right = st.columns(2)
    with left:
        tenure = st.slider('Months as a customer', 0, 72, 12)
        monthly = st.number_input('Monthly charges ($)', 0.0, 200.0, 70.0)
        total = st.number_input('Total charges to date ($)', 0.0, 20000.0, 840.0)
        contract = st.selectbox('Contract', ['Month-to-month', 'One year', 'Two year'])
    with right:
        internet = st.selectbox('Internet service', ['DSL', 'Fiber optic', 'No'])
        payment = st.selectbox('Payment method', ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])
        paperless = st.selectbox('Paperless billing', ['Yes', 'No'])
    submitted = st.form_submit_button('Estimate churn')
if submitted:
    row = pd.DataFrame([{'tenure': tenure, 'MonthlyCharges': monthly, 'TotalCharges': total, 'Contract': contract, 'InternetService': internet, 'PaymentMethod': payment, 'PaperlessBilling': paperless}])
    probability = load_model().predict_proba(row)[0, 1]
    st.metric('Estimated probability of leaving', f'{probability:.1%}')
    st.write('Predicted class at a 50% threshold: **' + ('Left' if probability >= .5 else 'Stayed') + '**')
    st.caption('Changing an input shows model behavior; it does not prove that changing that factor prevents churn.')
with st.expander('Model evaluation'):
    path = ROOT / 'reports/metrics.json'
    if path.exists():
        summary = json.loads(path.read_text())
        st.write('Selected using training cross-validation:', summary['selected_model'])
        st.image(str(ROOT / 'reports/confusion_matrix.png'))
        st.json(summary)
