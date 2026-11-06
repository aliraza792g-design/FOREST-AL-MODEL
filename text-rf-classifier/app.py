"""Streamlit app: run with `streamlit run app.py`."""
import json
import os

import joblib
import pandas as pd
import streamlit as st

from text_utils import clean_text

st.set_page_config(page_title="Text Classifier", page_icon="📝", layout="wide")
st.title("📝 Random Forest Text Classifier")

MODEL_PATH = "models/text_rf.joblib"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


if not os.path.exists(MODEL_PATH):
    st.error("Model not found. Run `python train_model.py --demo` first "
             "and commit the `models/` folder.")
    st.stop()

model = load_model()
tab1, tab2 = st.tabs(["Predict", "Model performance"])

with tab1:
    text = st.text_area("Enter text to classify", height=180,
                        placeholder="Paste a sentence, review, or article here...")
    if st.button("Classify", type="primary") and text.strip():
        cleaned = clean_text(text)
        pred = model.predict([cleaned])[0]
        proba = model.predict_proba([cleaned])[0]
        st.success(f"Predicted class: **{pred}**")
        st.bar_chart(pd.Series(proba, index=model.classes_, name="probability"))

with tab2:
    if os.path.exists("outputs/metrics.json"):
        m = json.load(open("outputs/metrics.json"))
        c = st.columns(4)
        c[0].metric("Accuracy", f"{m['accuracy']:.3f}")
        c[1].metric("Precision", f"{m['precision_weighted']:.3f}")
        c[2].metric("Recall", f"{m['recall_weighted']:.3f}")
        c[3].metric("F1", f"{m['f1_weighted']:.3f}")
        st.caption(f"Best params: {m['best_params']}")
    if os.path.exists("outputs/classification_report.txt"):
        st.code(open("outputs/classification_report.txt").read())
    a, b = st.columns(2)
    if os.path.exists("outputs/confusion_matrix.png"):
        a.image("outputs/confusion_matrix.png", caption="Confusion matrix")
    if os.path.exists("outputs/feature_importance.png"):
        b.image("outputs/feature_importance.png", caption="Top 20 features")
