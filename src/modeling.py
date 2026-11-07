"""Pipelines e treino dos modelos de classificacao."""
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from preprocessing import log_skewed, add_features

SEED = 42
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)


def build_models(use_fe=True):
    pre = [("fe", FunctionTransformer(add_features))] if use_fe else []
    pre += [("log", FunctionTransformer(log_skewed)), ("scale", StandardScaler())]
    lr = Pipeline(pre + [("clf", LogisticRegression(class_weight="balanced", max_iter=2000))])
    rf = Pipeline(pre + [("clf", RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample", random_state=SEED, n_jobs=-1))])
    return {
        "Regressao Logistica": (lr, {"clf__C": [0.01, 0.1, 1, 10]}),
        "Random Forest": (rf, {"clf__max_depth": [None, 8], "clf__min_samples_leaf": [1, 3]}),
    }


def tune(grids, X_train, y_train, scoring="average_precision"):
    best = {}
    for name, (pipe, grid) in grids.items():
        gs = GridSearchCV(pipe, grid, scoring=scoring, cv=CV, n_jobs=-1)
        gs.fit(X_train, y_train)
        best[name] = gs
    return best
