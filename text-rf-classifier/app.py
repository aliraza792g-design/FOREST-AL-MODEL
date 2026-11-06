"""Streamlit app: run with `streamlit run app.py`."""
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from text_utils import clean_text

st.set_page_config(page_title="Text Classifier", page_icon="📝", layout="wide")
st.title("📝 Random Forest Text Classifier")

BASE = Path(__file__).resolve().parent  # works even if app is in a subfolder
MODEL_PATH = BASE / "models" / "text_rf.joblib"
OUT = BASE / "outputs"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


if not MODEL_PATH.exists():
    st.error(f"Model not found at {MODEL_PATH}. Run `python train_model.py --demo` "
             "and commit the `models/` folder.")
    st.write("Files next to app.py:", sorted(p.name for p in BASE.iterdir()))
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
    if (OUT / "metrics.json").exists():
        m = json.load(open(OUT / "metrics.json"))
        c = st.columns(4)
        c[0].metric("Accuracy", f"{m['accuracy']:.3f}")
        c[1].metric("Precision", f"{m['precision_weighted']:.3f}")
        c[2].metric("Recall", f"{m['recall_weighted']:.3f}")
        c[3].metric("F1", f"{m['f1_weighted']:.3f}")
        st.caption(f"Best params: {m['best_params']}")
    if (OUT / "classification_report.txt").exists():
        st.code(open(OUT / "classification_report.txt").read())
    a, b = st.columns(2)
    if (OUT / "confusion_matrix.png").exists():
        a.image(str(OUT / "confusion_matrix.png"), caption="Confusion matrix")
    if (OUT / "feature_importance.png").exists():
        b.image(str(OUT / "feature_importance.png"), caption="Top 20 features")
