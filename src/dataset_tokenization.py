import sys
import os
from typing import Any
import pickle

from scipy import sparse

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk

nltk.download("stopwords")

from nltk.corpus import stopwords


DATA_DIR = "data/"
DATA_STORAGE = os.path.join(DATA_DIR, "data-storage")
FEATURE_STORAGE = os.path.join(DATA_DIR, "feature-storage")
TOKENIZER_DIRECTORY = "factorizer/"


def load_data_from_file(filename: str) -> pd.DataFrame:
    file_path = os.path.join(DATA_STORAGE, filename)
    return pd.read_csv(file_path)


def write_data_to_target_file(
    target_filename: str, features: sparse.csr_matrix
) -> None:
    try:
        os.makedirs(FEATURE_STORAGE)
    except FileExistsError:
        pass

    target_path = os.path.join(FEATURE_STORAGE, target_filename)
    with open(target_path, "wb") as f:
        sparse.save_npz(f, features, compressed=False)


def save_tokenizer(tokenizer: Any) -> None:
    try:
        os.makedirs(TOKENIZER_DIRECTORY)
    except FileExistsError:
        pass

    obj_path = os.path.join(TOKENIZER_DIRECTORY, "tfidf_vectorizer.pickle")
    with open(obj_path, "wb") as f:
        pickle.dump(tokenizer, f, protocol=pickle.HIGHEST_PROTOCOL)


def normalize_data_pipeline(filename: str, target_filename: str) -> None:
    df_train = load_data_from_file("train__" + filename).fillna("")
    df_test = load_data_from_file("test__" + filename).fillna("")
    stopwords_list = stopwords.words("russian")
    tf_idf_vectorizer = TfidfVectorizer(
        sublinear_tf=True,
        strip_accents="unicode",
        analyzer="char",
        stop_words=stopwords_list,
        ngram_range=(2, 12),
        max_features=10000,
    )
    tf_idf_vectorizer.fit(df_train["comment"].append(df_test["comment"]))

    features_train = tf_idf_vectorizer.transform(df_train["comment"])
    features_test = tf_idf_vectorizer.transform(df_test["comment"])

    write_data_to_target_file("train__" + target_filename, features_train)
    write_data_to_target_file("test__" + target_filename, features_test)

    save_tokenizer(tf_idf_vectorizer)


if __name__ == "__main__":
    normalize_data_pipeline(sys.argv[1], sys.argv[2])
