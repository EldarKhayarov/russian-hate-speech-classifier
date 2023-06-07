import re

from src.dataset_preprocessing import normalize_phrase_fab


def preprocess_phrase(phrase: str) -> str:
    phrase = phrase.lower()
    phrase = re.sub(r"([^ а-яё])|(\d)", " ", phrase)
    phrase = re.sub(r"\s{2,}", " ", phrase)
    phrase = re.sub(r"(^ )|( $)", "", phrase)
    phrase = normalize_phrase_fab()(phrase)
    return phrase


async def make_prediction(factorizer, model, comment: str) -> float:
    comment = preprocess_phrase(comment)

    features = factorizer.transform([comment])
    target_values = model.predict(features)
    return target_values[0]
