import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import spacy
from sentence_transformers import SentenceTransformer

from .common import (
    PROCESSED_DATA_PATH,
    RAW_INPUT_FILE,
    SBERT_MODEL_DIR,
    ensure_sbert_model,
    normalize_storage_label,
)


DEFAULT_INPUT_FILE = RAW_INPUT_FILE
DEFAULT_OUTPUT_FILE = PROCESSED_DATA_PATH
DEFAULT_MAX_TOKENS = 350

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
    parser = argparse.ArgumentParser(description="Preprocess texts into SBERT embeddings.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_FILE, help="Path to the source Excel file.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_FILE, help="Path to the output JSON file.")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS, help="Max tokens per segment.")
    parser.add_argument(
        "--device",
        default="cuda",
        help="SentenceTransformer device, for example 'cuda' or 'cpu'.",
    )
    return parser.parse_args()


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


def build_segments(text, max_tokens, tokenizer):
    sentences = re.split(r"(?<=[.!?])\s+", text.replace("\n", " ").strip())
    segments = []
    buffer = ""

    for sentence in sentences:
        sentence_tokens = tokenizer.tokenize(sentence)
        if len(sentence_tokens) > max_tokens:
            for start in range(0, len(sentence_tokens), max_tokens):
                chunk = sentence_tokens[start:start + max_tokens]
                segments.append(tokenizer.convert_tokens_to_string(chunk))
            continue

        candidate = f"{buffer} {sentence}".strip() if buffer else sentence
        if len(tokenizer.tokenize(candidate)) > max_tokens:
            if buffer:
                segments.append(buffer)
            buffer = sentence
        else:
            buffer = candidate

    if buffer:
        segments.append(buffer)
    return segments


def main():
    args = parse_args()
    input_path = args.input.resolve()
    output_path = args.output.resolve()

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {input_path}. If you only want to run the existing models, "
            "you can skip preprocessing because processed_final.json is already included."
        )

    ensure_sbert_model()
    model = SentenceTransformer(str(SBERT_MODEL_DIR), device=args.device)
    tokenizer = model.tokenizer
    nlp = spacy.load("en_core_web_sm")

    df = pd.read_excel(input_path, engine="openpyxl")
    df["text"] = df["text"].fillna("")
    df["source"] = df["source"].fillna("")

    final_data = []
    for row_id, row in df.iterrows():
        clean_text = preprocess_text(row["text"])
        clean_text = conditional_replace(clean_text, row["source"])
        masked_text = mask_named_entities(clean_text, nlp)
        segments = build_segments(masked_text, args.max_tokens, tokenizer)

        if not segments:
            continue

        embeddings = model.encode(segments, show_progress_bar=False)
        mean_embedding = np.mean(embeddings, axis=0).tolist()
        final_data.append(
            {
                "idx": int(row_id),
                "label": normalize_storage_label(row["source"]),
                "segments": segments,
                "embedding": mean_embedding,
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(final_data, f, ensure_ascii=False, indent=2)

    print(f"Finished. Output: {output_path}")
    print(f"Samples written: {len(final_data)}")


if __name__ == "__main__":
    main()
