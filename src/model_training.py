import sys
import os
import pickle
import pandas as pd
from typing import Iterable

import numpy as np
from scipy import sparse

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier


DATA_DIR = "data/"
DATA_STORAGE = os.path.join(DATA_DIR, "data-storage")
FEATURE_STORAGE = os.path.join(DATA_DIR, "feature-storage")
TRAIN_STORAGE = os.path.join(DATA_DIR, "train-storage")
ML_MODELS = {
    "svc": SVC,
    "logistic_regression": LogisticRegression,
    "decision_tree": DecisionTreeClassifier,
    "random_forest": RandomForestClassifier,
    "xgboost": XGBClassifier,
}


def camel_to_kebab(raw_string: str) -> str:
    return raw_string.replace("_", "-")


def get_model(model_name: str):
    return ML_MODELS[model_name]()


def train_model(model_name: str, features: Iterable, target_values: Iterable):
    model = get_model(model_name)
    model.fit(features, target_values)
    return model


def load_data_from_file(filename: str) -> pd.DataFrame:
    file_path = os.path.join(DATA_STORAGE, filename)
    return pd.read_csv(file_path)


def load_features(features_filename: str) -> Iterable:
    target_path = os.path.join(FEATURE_STORAGE, features_filename)

    with open(target_path, "rb") as f:
        features = sparse.load_npz(f)

    return features


def save_model(path: str, model) -> None:
    try:
        os.makedirs(os.path.dirname(path))
    except FileExistsError:
        pass

    with open(path, "wb") as f:
        pickle.dump(model, f)


def save_predictions(path: str, predictions: np.array) -> None:
    try:
        os.makedirs(os.path.dirname(path))
    except FileExistsError:
        pass

    with open(path, "wb") as f:
        np.save(f, predictions)


def train_model_pipeline(
    data_filename: str, features_filename: str, model_name: str
) -> None:
    features = load_features(features_filename)
    marked_data = load_data_from_file(data_filename)
    target_values = marked_data["toxic"]
    trained_model = train_model(model_name, features, target_values)

    y_predicted = trained_model.predict(features)

    train_directory_path = os.path.join(TRAIN_STORAGE, camel_to_kebab(model_name))
    save_model(os.path.join(train_directory_path, "model.pickle"), trained_model)
    save_predictions(os.path.join(train_directory_path, "predictions.npy"), y_predicted)


if __name__ == "__main__":
    train_model_pipeline(sys.argv[1], sys.argv[2], sys.argv[3])
