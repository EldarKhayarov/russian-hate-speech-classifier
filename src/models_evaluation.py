import sys
import os
import json
import logging
from typing import Optional

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

logger = logging.getLogger("models evaluation")
logger.setLevel(logging.INFO)


def camel_to_kebab(raw_string: str) -> str:
    return raw_string.replace("_", "-")


def load_data_from_file(filename: str) -> pd.DataFrame:
    file_path = os.path.join(DATA_STORAGE, filename)
    return pd.read_csv(file_path)


def load_predicted_values(path: str) -> np.array:
    with open(path, "rb") as f:
        array = np.load(f)

    return array


def read_evaluation_results() -> Optional[dict]:
    try:
        os.makedirs(EVALUATION_STORAGE)
    except FileExistsError:
        pass
    else:
        return

    with open(os.path.join(EVALUATION_STORAGE, "results.json"), "r") as f:
        results = json.load(f)

    return results


def save_evaluation_results(evaluations: dict) -> None:
    try:
        os.makedirs(EVALUATION_STORAGE)
    except FileExistsError:
        pass

    with open(os.path.join(EVALUATION_STORAGE, "results.json"), "w") as f:
        json.dump(evaluations, f)


def add_best_tag_to_model(model_name: str) -> None:
    model_dir = os.path.join(TRAIN_STORAGE, camel_to_kebab(model_name))
    os.rename(
        os.path.join(model_dir, "model.pickle"),
        os.path.join(model_dir, "best__model.pickle"),
    )


def calculate_f1(model_name: str, true_values: pd.Series):
    predictions_path = os.path.join(
        TRAIN_STORAGE, camel_to_kebab(model_name), "predictions.npy"
    )
    predictions = load_predicted_values(predictions_path)
    f1 = f1_score(true_values, predictions)
    return f1


def model_eval_pipeline(data_filename: str) -> None:
    test_data = load_data_from_file("test__" + data_filename)
    true_values = test_data["toxic"]

    last_results = read_evaluation_results()

    new_results = {}
    best_f1 = 0
    best_f1_model_name = None

    for model_name in MODELS:
        new_f1 = calculate_f1(model_name, true_values)
        f1_to_write = new_f1

        if last_results and last_results.get(model_name):
            old_result_f1 = last_results[model_name].get("f1", 0.0)
            if new_f1 > old_result_f1:
                # update best model
                add_best_tag_to_model(model_name)
                logger.info(
                    f"Model `{model_name}` has been updated to the best version [F1: {old_result_f1} -> {new_f1}]"
                )

            else:
                f1_to_write = old_result_f1

        else:
            add_best_tag_to_model(model_name)

        new_results[model_name] = {
            "f1": f1_to_write,
        }

        if new_f1 > best_f1:
            best_f1 = new_f1
            best_f1_model_name = model_name

    f1_to_write = best_f1
    if last_results and last_results.get("__BEST_MODEL"):
        old_result_f1 = last_results["__BEST_MODEL"].get("f1", 0.0)
        if best_f1 > old_result_f1:
            logger.info(
                f"BEST model [`{last_results['__BEST_MODEL']}` -> `{best_f1_model_name}`] "
                f"has been updated to the best version [F1: {old_result_f1} -> {best_f1}]"
            )
        else:
            f1_to_write = old_result_f1

    new_results["__BEST_MODEL"] = {"model_name": best_f1_model_name, "f1": f1_to_write}

    save_evaluation_results(new_results)


if __name__ == "__main__":
    model_eval_pipeline(sys.argv[1])
