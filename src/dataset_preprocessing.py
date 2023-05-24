import sys
import os

import pandas as pd
import nltk

nltk.download("stopwords")

from nltk.corpus import stopwords
from pymorphy2 import MorphAnalyzer


DATA_DIR = "data/"
SOURCE_DIR = os.path.join(DATA_DIR, "raw-source")
TARGET_DIR = os.path.join(DATA_DIR, "data-storage")


def normalize_phrase_fab():
    stopwords_list = stopwords.words("russian")
    morph_analyzer = MorphAnalyzer()

    def normalize_phrase(phrase: str) -> str:
        phrase = " ".join(
            [
                morph_analyzer.normal_forms(word)[0]
                for word in phrase.split()
                if phrase not in stopwords_list
            ]
        )
        return phrase

    return normalize_phrase


def clear_phrases(collection: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    collection.replace(r"([^ А-ЯЁа-яё])|(\d)", " ", regex=True, inplace=True)
    collection.replace(r"\s{2,}", " ", regex=True, inplace=True)
    collection.replace(r"(^ )|( $)", "", regex=True, inplace=True)
    return collection


def load_data_from_file(filename: str) -> pd.DataFrame:
    file_path = os.path.join(SOURCE_DIR, filename)
    return pd.read_csv(file_path)


def write_data_to_target_file(target_filename: str, data: pd.DataFrame) -> None:
    try:
        os.makedirs(TARGET_DIR)
    except FileExistsError:
        pass

    target_path = os.path.join(TARGET_DIR, target_filename)
    data.to_csv(target_path, index=False)


def normalize_data_pipeline(filename: str, target_filename: str) -> None:
    df = load_data_from_file(filename)
    X = clear_phrases(df["comment"])
    X = X.str.lower()
    X = X.apply(normalize_phrase_fab())
    df["comment"] = X
    df.dropna(inplace=True)
    write_data_to_target_file(target_filename, df)


if __name__ == "__main__":
    normalize_data_pipeline(sys.argv[1], sys.argv[2])
