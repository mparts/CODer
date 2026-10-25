"""
Baseline classifiers (scikit-learn).
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from code_origin.config import BASE_MODELS, SEED, REG_STRENGTH, MAX_ITER, LR_CLASS_WEIGHTS


def show_whitespace(text): # the char vectorizer collapses runs of whitespace, so make layout visible first
    return text.replace("\n", "¶").replace("\t", "→").replace(" ", "·")


def make_baseline(name, seed=SEED):
    lr = LogisticRegression(C=REG_STRENGTH, max_iter=MAX_ITER, class_weight=LR_CLASS_WEIGHTS, random_state=seed)
    if name == BASE_MODELS[0]: # code tokens only, ignores layout
        vec = TfidfVectorizer(token_pattern=r"\w+|[^\w\s]", ngram_range=(1, 3), min_df=2, sublinear_tf=True, lowercase=False)
    elif name == BASE_MODELS[1]: # character n-grams, sees indentation, newlines and spacing
        vec = TfidfVectorizer(analyzer="char", ngram_range=(1, 6), min_df=2, sublinear_tf=True, preprocessor=show_whitespace, max_features=300_000)
    else:
        raise ValueError(f"unknown model {name}")
    return make_pipeline(vec, lr)