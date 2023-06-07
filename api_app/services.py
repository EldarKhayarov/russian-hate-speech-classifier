import re

import nltk

nltk.download("stopwords")

from nltk.corpus import stopwords
from pymorphy2 import MorphAnalyzer


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


def preprocess_phrase(phrase: str) -> str:
    phrase = phrase.lower()
    phrase = re.sub(r"([^ а-яё])|(\d)", " ", phrase)
    phrase = re.sub(r"\s{2,}", " ", phrase)
    phrase = re.sub(r"(^ )|( $)", "", phrase)
    phrase = normalize_phrase(phrase)
    return phrase


async def make_prediction(factorizer, model, comment: str) -> float:
    comment = preprocess_phrase(comment)

    features = factorizer.transform([comment])
    target_values = model.predict(features)
    return target_values[0]
