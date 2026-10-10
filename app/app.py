"""Streamlit front end: loads models/model.pkl and predicts a high-risk dengue month.

Run from the repository root:
    streamlit run app/app.py
"""
import calendar
import pickle
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from features import build_features  # noqa: E402

st.set_page_config(page_title="Dengue Risk Predictor — Sri Lanka", page_icon="🦟", layout="centered")

@st.cache_resource
def load_bundle():
    with open(ROOT / "models" / "model.pkl", "rb") as f:
        return pickle.load(f)


bundle = load_bundle()
model, thresholds = bundle["model"], bundle["thresholds"]

st.title("🦟 Dengue High-Risk Month Predictor")
st.caption(
    f"Sri Lanka, 26 districts / RDHS areas · {bundle['model_name']} trained on {bundle['train_years']} · "
    f"test F1 {bundle['test_metrics']['F1']:.2f}, ROC-AUC {bundle['test_metrics']['ROC-AUC']:.2f}"
)
st.write(
    "Enter the recent dengue case counts for a district. The model estimates the probability that the "
    "coming month will be a **high-risk month**, meaning cases above that district's usual 75th percentile."
)

with st.form("inputs"):
    c1, c2 = st.columns(2)
    district = c1.selectbox("District / RDHS area", bundle["districts"], index=bundle["districts"].index("Colombo"))
    month = c2.selectbox("Month to predict", list(range(1, 13)), index=11, format_func=lambda m: calendar.month_name[m])

    st.markdown("**Reported cases in previous months**")
    c3, c4, c5 = st.columns(3)
    lag1 = c3.number_input("Last month", min_value=0, value=59, step=1)
    lag2 = c4.number_input("2 months ago", min_value=0, value=151, step=1)
    lag3 = c5.number_input("3 months ago", min_value=0, value=192, step=1)
    lag12 = st.number_input("Same month last year", min_value=0, value=3283, step=1)
    submitted = st.form_submit_button("Predict", type="primary", use_container_width=True)

if submitted:
    raw = pd.DataFrame([{"district": district, "month": month, "lag1": lag1, "lag2": lag2, "lag3": lag3, "lag12": lag12}])
    prob = float(model.predict_proba(build_features(raw))[0, 1])
    high = prob >= 0.5
    thr = thresholds[district]

    if high:
        st.error(f"### HIGH RISK — {prob:.0%} probability")
    else:
        st.success(f"### Normal — {prob:.0%} probability of a high-risk month")
    st.progress(prob)

    m1, m2, m3 = st.columns(3)
    m1.metric("High-risk threshold", f"{thr:,.0f} cases")
    m2.metric("Last month", f"{lag1:,}", delta=f"{lag1 - lag2:+,} vs month before", delta_color="inverse")
    m3.metric("3-month average", f"{(lag1 + lag2 + lag3) / 3:,.0f}")

    with st.expander("Features sent to the model"):
        feats = build_features(raw).iloc[0]
        st.dataframe(pd.DataFrame({"feature": feats.index, "value": feats.astype(str).values}), hide_index=True)

st.divider()
st.caption(
    "Data: Epidemiology Unit, Ministry of Health Sri Lanka (monthly cases 2010–2020, via Kaggle). "
    "Coursework demo, not for operational public-health decisions."
)
