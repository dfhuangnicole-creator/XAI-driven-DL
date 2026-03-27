import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_TFIDF_DIR = DATA_DIR / "processed" / "tfidf"
DEFAULT_INPUT_FILE = RAW_DATA_DIR / "data.xlsx"
DEFAULT_OUTPUT_FILE = PROCESSED_TFIDF_DIR / "processed_TFIDF_final.json"
DEFAULT_FEATURES_FILE = PROCESSED_TFIDF_DIR / "feature_names.json"

CONTRACTIONS = {
    "don't": "do not", "doesn't": "does not", "isn't": "is not",
    "aren't": "are not", "wasn't": "was not", "weren't": "were not",
    "haven't": "have not", "hasn't": "has not", "hadn't": "had not",
    "won't": "will not", "wouldn't": "would not", "can't": "can not",
    "couldn't": "could not", "shouldn't": "should not", "mightn't": "might not",
    "mustn't": "must not", "it's": "it is", "i'm": "i am", "he's": "he is",
    "she's": "she is", "that's": "that is", "what's": "what is", "there's": "there is",
    "who's": "who is", "let's": "let us"
}

NON_TRANSLATED_WORDS = [w.lower() for w in [
    "Russia", "Ukraine", "UN", "Council", "humanitarian", "was", "States", "Thank", "United", "President",
    "peace", "support", "security", "continue", "had", "civilians", "efforts", "conflict", "DPRK", "violence",
    "Security", "thank", "international", "attacks", "regime", "Colleagues", "including", "Gaza", "war", "General",
    "weapons", "Israel", "Syria", "Russian", "Sudan", "Secretary", "assistance", "Madam", "were", "briefing",
    "Ukrainian", "Hamas", "Resolution", "aid", "urge", "Iran", "critical", "continues", "Member", "committed",
    "Chinese", "call", "Palestinian", "welcome", "accountability", "Syrian", "Haiti", "partners", "civilian", "address",
    "mandate", "parties", "Charter", "Libya", "violations", "rights", "DRC", "forces", "ensure", "OPCW", "resolution",
    "today", "Yemen", "abuses", "access", "Assad", "clear", "Representative", "actions", "said", "Houthis", "chemical",
    "man", "ballistic", "justice", "ceasefire", "resolutions", "crisis", "Putin", "work", "regional", "remains",
    "Envoy", "people", "mission", "region", "Palestinians", "Special", "armed", "need", "SRSG", "ISIS", "briefings",
    "ongoing", "elections", "missile", "commitment", "progress", "atrocities", "also", "crimes", "then", "aggression",
    "African", "community", "Libyan", "violation", "hostages", "condemn", "situation", "Sudanese", "implement",
    "sanctions", "Kremlin", "actors", "inclusive", "accountable", "Israeli", "stability", "peacekeepers", "MONUSCO",
    "threat", "Biden", "infrastructure", "Taliban", "Iraq", "challenges", "implementation", "has", "commend", "reports",
    "insecurity", "personnel", "displaced", "terrorist", "food", "supporting", "global", "reiterate", "sexual", "Afghan",
    "groups", "missiles", "invasion", "nuclear", "human", "came", "refugees", "went", "be"
]]

TRANSLATED_WORDS = [w.lower() for w in [
    "international", "countries", "security", "was", "humanitarian", "peace", "President", "Council", "China",
    "parties", "UN", "community", "efforts", "situation", "support", "had", "Security", "conflict", "Gaza",
    "regional", "dialogue", "Sudan", "development", "were", "Ukraine", "thank", "political", "crisis",
    "cooperation", "stability", "Africa", "ceasefire", "terrorism", "issue", "Palestinian", "sanctions",
    "continue", "resolution", "Thank", "General", "solution", "assistance", "Israel", "African", "Haiti",
    "concerned", "nuclear", "supports", "settlement", "promote", "Secretary", "civilians", "relevant", "region",
    "role", "Madam", "briefings", "said", "constructive", "mandate", "actions", "Representative", "DPRK",
    "implementation", "challenges", "Peninsula", "strengthen", "effectively", "Syria", "hope", "DRC", "concerns",
    "issues", "welcome", "operations", "Palestine", "promoting", "sustainable", "weapons", "global", "Haitian",
    "resolutions", "has", "briefing", "sovereignty", "country", "Resolution", "non", "draft", "Somalia", "calls",
    "welcomes", "consensus", "tensions", "counter", "peacekeeping", "capacity", "actively", "process", "conflicts",
    "Special", "create", "lasting", "meeting", "talks", "Third", "implement", "call", "maintain", "Envoy",
    "terrorist", "Afghanistan", "unilateral", "Second", "Afghan", "common", "violence", "Yemen", "casualties", "then"
]]

PAT_NON_TRANSLATED = re.compile(
    r"\b(" + "|".join(map(re.escape, NON_TRANSLATED_WORDS)) + r")\b",
    flags=re.IGNORECASE,
)
PAT_TRANSLATED = re.compile(
    r"\b(" + "|".join(map(re.escape, TRANSLATED_WORDS)) + r")\b",
    flags=re.IGNORECASE,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess texts into TF-IDF features.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_FILE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_FILE)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES_FILE)
    return parser.parse_args()


def normalize_storage_label(value):
    if isinstance(value, (int, float)):
        numeric = int(value)
        if numeric in (1, 2):
            return numeric
        if numeric in (0, 1):
            return numeric + 1

    normalized = str(value).strip().lower().replace("_", " ").replace("-", " ")
    normalized = " ".join(normalized.split())
    mapping = {
        "non translated": 1,
        "non translated e": 1,
        "nontranslated": 1,
        "nontranslated e": 1,
        "translated": 2,
        "translated e": 2,
    }
    if normalized in mapping:
        return mapping[normalized]

    raise ValueError(f"Unsupported label value: {value!r}")


def expand_contractions(text, contractions_dict):
    pattern = re.compile(
        "({})".format("|".join(re.escape(key) for key in contractions_dict.keys())),
        flags=re.IGNORECASE,
    )

    def replace(match):
        contraction = match.group(0)
        expanded = contractions_dict[contraction.lower()]
        if contraction[0].isupper():
            return expanded[0].upper() + expanded[1:]
        return expanded

    return pattern.sub(replace, text)


def preprocess_text(text):
    if not isinstance(text, str):
        text = str(text)
    text = expand_contractions(text, CONTRACTIONS)
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def conditional_replace(text, label):
    normalized_label = str(label).strip().lower()
    if normalized_label == "non-translated e":
        return PAT_NON_TRANSLATED.sub("UNTOKEN", text)
    if normalized_label == "translated e":
        return PAT_TRANSLATED.sub("UNTOKEN", text)
    return text


def mask_named_entities(text, nlp):
    doc = nlp(text)
    allowed_entities = {"PERSON", "ORG", "GPE", "LOC", "NORP", "FAC"}
    return " ".join(
        f"<{token.ent_type_}>" if token.ent_type_ in allowed_entities else token.text
        for token in doc
    )


def main():
    args = parse_args()
    input_path = args.input.resolve()
    output_path = args.output.resolve()
    features_path = args.features.resolve()

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {input_path}. If you only want to run the baseline classifier, "
            "you can reuse the existing processed_TFIDF_final.json."
        )

    nlp = spacy.load("en_core_web_sm")
    df = pd.read_excel(input_path, engine="openpyxl")
    df["text"] = df["text"].fillna("")
    df["source"] = df["source"].fillna("")

    processed_texts = []
    labels = []
    for _, row in df.iterrows():
        text = preprocess_text(row["text"])
        text = conditional_replace(text, row["source"])
        text = mask_named_entities(text, nlp)
        processed_texts.append(text)
        labels.append(normalize_storage_label(row["source"]))

    tfidf = TfidfVectorizer(max_features=4096, ngram_range=(1, 3), stop_words=None)
    tfidf_vectors = tfidf.fit_transform(processed_texts).toarray()

    final_data = []
    for idx, (text, label, embedding) in enumerate(zip(processed_texts, labels, tfidf_vectors)):
        final_data.append(
            {
                "idx": idx,
                "label": label,
                "text": text,
                "embedding": embedding.tolist(),
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(final_data, f, ensure_ascii=False, indent=2)
    print(f"Finished. Output: {output_path}")

    total_elements = tfidf_vectors.size
    nonzero_elements = np.count_nonzero(tfidf_vectors)
    sparsity = 1 - (nonzero_elements / total_elements)
    print(f"TF-IDF matrix sparsity: {sparsity:.4f}")

    feature_names = tfidf.get_feature_names_out()
    with features_path.open("w", encoding="utf-8") as f:
        json.dump(feature_names.tolist(), f, ensure_ascii=False, indent=2)
    print(f"TF-IDF feature names saved to: {features_path}")


if __name__ == "__main__":
    main()
