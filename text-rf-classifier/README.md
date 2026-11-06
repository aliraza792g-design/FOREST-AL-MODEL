# Random Forest Text Classifier

TF-IDF + RandomForest text classification with a Streamlit demo.

## Run locally
```bash
pip install -r requirements.txt
python train_model.py --demo                       # demo data (20 Newsgroups)
# or your own data:
python train_model.py --data data/my.csv --text-col text --label-col label
streamlit run app.py
```

## Deploy
1. Push this folder to GitHub (include `models/` and `outputs/`).
2. Go to share.streamlit.io -> New app -> pick repo, branch, main file `app.py`.
