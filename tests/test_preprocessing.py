"""
Unit Tests for NLP Preprocessing Pipeline.
Tests tokenization, normalization, punctuation removal, stopword retention, and lemmatization.
"""

import pytest
from nlp.preprocessing import TextPreprocessor, preprocess_text, tokenize_text


@pytest.fixture
def preprocessor():
    return TextPreprocessor(use_lemmatization=True, remove_stopwords=True)


def test_clean_text_basic(preprocessor):
    raw = "Hello, World! Welcome to NLP 101."
    cleaned = preprocessor.clean_text(raw)
    assert "hello" in cleaned
    assert "world" in cleaned
    assert "nlp 101" in cleaned
    assert "!" not in cleaned
    assert "," not in cleaned


def test_clean_text_removes_urls_and_emails(preprocessor):
    raw = "Visit https://example.com or mail test@university.edu for AI info."
    cleaned = preprocessor.clean_text(raw)
    assert "https" not in cleaned
    assert "example" not in cleaned
    assert "test@university.edu" not in cleaned
    assert "ai info" in cleaned


def test_clean_text_empty_and_whitespace(preprocessor):
    assert preprocessor.clean_text("") == ""
    assert preprocessor.clean_text("   ") == ""
    assert preprocessor.clean_text(None) == ""


def test_tokenize(preprocessor):
    text = "Machine learning algorithms process raw data."
    tokens = preprocessor.tokenize(text)
    assert isinstance(tokens, list)
    assert len(tokens) == 6
    assert "machine" in tokens
    assert "learning" in tokens


def test_lemmatization(preprocessor):
    token = "running"
    lemmatized = preprocessor.lemmatize_token(token)
    assert lemmatized == "run"

    token_plural = "networks"
    lemmatized_plural = preprocessor.lemmatize_token(token_plural)
    assert lemmatized_plural == "network"


def test_conversational_stopword_retention(preprocessor):
    # 'what', 'how' must NOT be dropped because they distinguish questions
    raw = "What is artificial intelligence?"
    processed = preprocessor.preprocess(raw)
    assert "what" in processed
    assert "artificial" in processed
    assert "intelligence" in processed


def test_special_characters_and_symbols(preprocessor):
    raw = "###$$$@@@ Python & ML @@@$$$###"
    processed = preprocessor.preprocess(raw)
    assert "python" in processed
    assert "ml" in processed
    assert "#" not in processed
    assert "$" not in processed


def test_convenience_functions():
    clean = preprocess_text("Hi there, how are you?")
    assert isinstance(clean, str)
    assert len(clean) > 0

    tokens = tokenize_text("Hello NLP world")
    assert isinstance(tokens, list)
    assert len(tokens) == 3
