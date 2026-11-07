"""Carga, limpeza e preparo dos dados do Wine Quality (WineQT)."""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

SKEWED = ["chlorides", "residual sugar", "sulphates", "total sulfur dioxide", "free sulfur dioxide"]


def load_data(path="../data/WineQT.csv"):
    return pd.read_csv(path)


def clean(df):
    """Remove Id e duplicatas (ignorando o Id) para evitar vazamento treino/teste."""
    df = df.drop(columns=["Id"], errors="ignore")
    return df.drop_duplicates().reset_index(drop=True)


def make_target(df, cutoff=7):
    """Alvo binario: 1 = alta qualidade (nota >= cutoff)."""
    df = df.copy()
    df["alta_qualidade"] = (df["quality"] >= cutoff).astype(int)
    return df


def add_features(X):
    """Feature engineering simples e interpretavel."""
    X = X.copy()
    X["so2_ratio"] = X["free sulfur dioxide"] / X["total sulfur dioxide"]
    X["total_acidity"] = X["fixed acidity"] + X["volatile acidity"]
    X["alcohol_x_sulphates"] = X["alcohol"] * X["sulphates"]
    return X


def log_skewed(X):
    """log1p nas variaveis de cauda longa (reduz efeito de outliers sem remove-los)."""
    X = X.copy()
    for c in SKEWED:
        X[c] = np.log1p(X[c])
    return X


def split(df, test_size=0.2, seed=42):
    X = df.drop(columns=["quality", "alta_qualidade"])
    y = df["alta_qualidade"]
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)
