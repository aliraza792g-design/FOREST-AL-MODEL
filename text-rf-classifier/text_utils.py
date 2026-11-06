"""Shared text preprocessing (used by both train_model.py and app.py)."""
import re
import string

import nltk

_READY = False
_STOP = None
_LEMMA = None


def ensure_nltk():
    """Download NLTK resources once (safe to call repeatedly)."""
    global _READY, _STOP, _LEMMA
    if _READY:
        return
    for pkg, path in [("stopwords", "corpora/stopwords"),
                      ("wordnet", "corpora/wordnet"),
                      ("omw-1.4", "corpora/omw-1.4")]:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(pkg, quiet=True)
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    _STOP = set(stopwords.words("english"))
    _LEMMA = WordNetLemmatizer()
    _READY = True


_PUNCT_TABLE = str.maketrans(string.punctuation, " " * len(string.punctuation))


def clean_text(text: str) -> str:
    """lowercase -> strip punctuation & numbers -> drop stop words -> lemmatize."""
    ensure_nltk()
    text = str(text).lower()
    text = text.translate(_PUNCT_TABLE)
    text = re.sub(r"\d+", " ", text)
    tokens = [_LEMMA.lemmatize(t) for t in text.split()
              if t not in _STOP and len(t) > 2]
    return " ".join(tokens)
