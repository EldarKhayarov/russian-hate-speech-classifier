import sys
import os
import json

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


DATA_DIR = "data/"
DATA_STORAGE = os.path.join(DATA_DIR, "data-storage")
TRAIN_STORAGE = os.path.join(DATA_DIR, "train-storage")
EVALUATION_STORAGE = os.path.join(DATA_DIR, "evaluation-storage")
MODELS = [
    "svc",
    "logistic_regression",
    "decision_tree",
    "random_forest",
    "xgboost",
]


def camel_to_kebab(raw_string: str) -> str:
    return raw_string.replace("_", "-")


def load_data_from_file(filename: str) -> pd.DataFrame:
    file_path = os.path.join(DATA_STORAGE, filename)
    return pd.read_csv(file_path)


def load_predicted_values(path: str) -> np.array:
    with open(path, "rb") as f:
        array = np.load(f)

    return array


def save_evaluation_results(evaluations: dict) -> None:
    try:
        os.makedirs(EVALUATION_STORAGE)
    except FileExistsError:
        pass

    with open(os.path.join(EVALUATION_STORAGE, "results.json"), "w") as f:
        json.dump(evaluations, f)


def calculate_f1(model_name: str, true_values: pd.Series):
    predictions_path = os.path.join(
        TRAIN_STORAGE, camel_to_kebab(model_name), "predictions.npy"
    )
    predictions = load_predicted_values(predictions_path)
    f1 = f1_score(true_values, predictions)
    return f1


def model_eval_pipeline(data_filename: str) -> None:
    marked_data = load_data_from_file(data_filename)
    true_values = marked_data["toxic"]
    model_f1_list = {}
    best_f1 = 0
    best_f1_model_name = None

    for model_name in MODELS:
        f1 = calculate_f1(model_name, true_values)
        model_f1_list[model_name] = {
            "f1": f1,
        }

        if f1 > best_f1:
            best_f1 = f1
            best_f1_model_name = model_name

    model_f1_list["__BEST_MODEL"] = {"model_name": best_f1_model_name, "f1": best_f1}
    save_evaluation_results(model_f1_list)


if __name__ == "__main__":
    model_eval_pipeline(sys.argv[1])
