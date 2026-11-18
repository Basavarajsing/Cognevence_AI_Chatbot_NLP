"""
Unit and Integration Tests for Chatbot Inference Engine.
Tests intent recognition, confidence scores, technical questions,
synonym questions, fallback triggering, and edge cases.
"""

import pytest
from nlp.chatbot import ChatbotEngine, get_chatbot_instance


@pytest.fixture(scope="module")
def bot():
    return get_chatbot_instance()


def test_bot_initialization(bot):
    assert bot.is_loaded is True
    assert len(bot.classes) > 0
    assert "greeting" in bot.classes
    assert "machine_learning" in bot.classes


def test_greeting_recognition(bot):
    queries = ["Hello", "Hi there", "Good morning", "Hey bot"]
    for q in queries:
        res = bot.get_response(q)
        assert res["intent"] == "greeting"
        assert res["confidence"] >= 0.25
        assert res["is_fallback"] is False
        assert len(res["response"]) > 0


def test_technical_questions_recognition(bot):
    # Test Machine Learning
    res_ml = bot.get_response("What is Machine Learning?")
    assert res_ml["intent"] in ["machine_learning", "artificial_intelligence"]
    assert res_ml["confidence"] >= 0.25

    # Test NLP
    res_nlp = bot.get_response("Explain Natural Language Processing")
    assert res_nlp["intent"] == "natural_language_processing"
    assert res_nlp["confidence"] >= 0.25

    # Test Python
    res_py = bot.get_response("Why is Python popular for data science?")
    assert res_py["intent"] in ["python_basics", "python_libraries", "data_science"]
    assert res_py["confidence"] >= 0.25


def test_different_ways_of_asking_same_intent(bot):
    # Multiple phrasings for goodbye
    goodbye_variants = [
        "Goodbye",
        "See you later",
        "I have to go now",
        "Bye bye"
    ]
    for variant in goodbye_variants:
        res = bot.get_response(variant)
        assert res["intent"] == "goodbye"

    # Multiple phrasings for thanks
    thanks_variants = [
        "Thank you so much",
        "Thanks for your help",
        "Much appreciated"
    ]
    for variant in thanks_variants:
        res = bot.get_response(variant)
        assert res["intent"] == "thanks"


def test_unknown_and_gibberish_triggers_fallback(bot):
    gibberish_queries = [
        "asdfghjklqwertyzxcvbnm",
        "zzz1239847120938471029384",
        "supercalifragilisticnonsense123"
    ]
    for q in gibberish_queries:
        res = bot.get_response(q)
        assert res["is_fallback"] is True
        assert res["intent"] == "fallback"


def test_empty_and_whitespace_input(bot):
    res_empty = bot.get_response("")
    assert res_empty["is_fallback"] is True
    assert res_empty["confidence"] == 0.0

    res_spaces = bot.get_response("     ")
    assert res_spaces["is_fallback"] is True
    assert res_spaces["confidence"] == 0.0


def test_long_input(bot):
    long_query = "What is artificial intelligence and how does machine learning work? " * 5
    res = bot.get_response(long_query)
    assert res["response"] is not None
    assert len(res["response"]) > 0


def test_special_characters_input(bot):
    res = bot.get_response("!@#$%^&*()_+{}[]:;<>?,./")
    assert res["is_fallback"] is True


def test_suggested_questions(bot):
    suggestions = bot.get_suggestions()
    assert isinstance(suggestions, list)
    assert len(suggestions) >= 4
    assert any("Artificial Intelligence" in s for s in suggestions)
