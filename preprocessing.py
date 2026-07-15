"""
preprocessing.py

Text cleaning and tokenization utilities shared by train_distilbert.py
(training time) and chatbot.py (inference time). Keeping this logic in one
module guarantees train/inference parity.
"""

import re
import unicodedata

_WHITESPACE_RE = re.compile(r"\s+")
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_MENTION_RE = re.compile(r"@\w+")
_MULTI_PUNCT_RE = re.compile(r"([!?.,])\1{1,}")


def clean_text(text: str) -> str:
    """Normalize raw user input before it reaches the tokenizer.

    Steps:
    1. Unicode-normalize (NFKC) to collapse visually-identical characters.
    2. Strip URLs and @mentions (not meaningful for intent classification).
    3. Collapse repeated punctuation ("!!!!" -> "!") which otherwise skews
       tokenization without adding semantic signal.
    4. Collapse whitespace and strip.
    """
    if not text:
        return ""

    text = unicodedata.normalize("NFKC", text)
    text = _URL_RE.sub(" ", text)
    text = _MENTION_RE.sub(" ", text)
    text = _MULTI_PUNCT_RE.sub(r"\1", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


def truncate_for_model(text: str, max_chars: int = 2000) -> str:
    """Hard safety cap on input length before it reaches any model call."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars]


def tokenize_for_classifier(text: str, tokenizer, max_length: int = 128):
    """Tokenize cleaned text using a HuggingFace tokenizer.

    Returns the dict of tensors produced by the tokenizer, ready to be fed
    into a DistilBERT-family model.
    """
    cleaned = clean_text(text)
    return tokenizer(
        cleaned,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
