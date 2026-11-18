"""
Text Preprocessing Pipeline for AI Chatbot.
Provides modular tokenization, normalization, punctuation removal,
and lemmatization designed for conversational intent classification.
"""

import re
import string
from typing import List

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# Ensure NLTK data availability with graceful fallbacks
try:
    nltk.data.find('corpora/wordnet')
    _lemmatizer = WordNetLemmatizer()
except Exception:
    try:
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
        _lemmatizer = WordNetLemmatizer()
    except Exception:
        _lemmatizer = None

try:
    nltk.data.find('tokenizers/punkt')
    _has_punkt = True
except Exception:
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
        _has_punkt = True
    except Exception:
        _has_punkt = False

# Conversational stop words: retain interrogatives, pronouns and negations
# which are essential for intent recognition in chat queries.
RETAIN_WORDS = {
    'what', 'how', 'who', 'why', 'where', 'which', 'when',
    'can', 'could', 'would', 'should', 'is', 'are', 'was', 'were',
    'do', 'does', 'did', 'you', 'your', 'i', 'my', 'me', 'we',
    'not', 'no', 'about', 'tell'
}

try:
    _base_stopwords = set(stopwords.words('english'))
    CHAT_STOPWORDS = _base_stopwords - RETAIN_WORDS
except Exception:
    CHAT_STOPWORDS = set()


class TextPreprocessor:
    """Modular text preprocessor for intent classification."""

    def __init__(self, use_lemmatization: bool = True, remove_stopwords: bool = True):
        self.use_lemmatization = use_lemmatization
        self.remove_stopwords = remove_stopwords
        self.lemmatizer = _lemmatizer if use_lemmatization else None

    def clean_text(self, text: str) -> str:
        """
        Normalize text: lowercase, remove non-printable chars,
        and trim excessive whitespace.
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Lowercase
        text = text.lower()
        
        # Replace URLs and emails with generic tokens if needed
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove special characters keeping alphanumeric and basic spacing
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        
        # Condense multiple whitespaces into a single space
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize normalized text into a list of words.
        Uses NLTK word_tokenize with regex fallback.
        """
        cleaned = self.clean_text(text)
        if not cleaned:
            return []

        if _has_punkt:
            try:
                tokens = nltk.word_tokenize(cleaned)
            except Exception:
                tokens = re.findall(r'\b\w+\b', cleaned)
        else:
            tokens = re.findall(r'\b\w+\b', cleaned)
            
        return tokens

    def lemmatize_token(self, token: str) -> str:
        """Lemmatize a single token using WordNetLemmatizer."""
        if not self.lemmatizer:
            return token
        try:
            # Lemmatize noun then verb form
            lemma = self.lemmatizer.lemmatize(token, pos='n')
            lemma = self.lemmatizer.lemmatize(lemma, pos='v')
            return lemma
        except Exception:
            return token

    def process_tokens(self, tokens: List[str]) -> List[str]:
        """Apply stopword filtering and lemmatization to token list."""
        processed = []
        for token in tokens:
            if not token or token in string.punctuation:
                continue
            if self.remove_stopwords and token in CHAT_STOPWORDS:
                continue
            if self.use_lemmatization:
                token = self.lemmatize_token(token)
            if token:
                processed.append(token)
        return processed

    def preprocess(self, text: str) -> str:
        """
        Full end-to-end preprocessing pipeline.
        Returns cleaned, normalized, lemmatized string suitable for TF-IDF.
        """
        tokens = self.tokenize(text)
        processed_tokens = self.process_tokens(tokens)
        return " ".join(processed_tokens)


# Singleton preprocessor for application-wide consistency
default_preprocessor = TextPreprocessor(use_lemmatization=True, remove_stopwords=True)


def preprocess_text(text: str) -> str:
    """Convenience function for preprocessing a raw query."""
    return default_preprocessor.preprocess(text)


def tokenize_text(text: str) -> List[str]:
    """Convenience function to tokenize a raw query."""
    return default_preprocessor.tokenize(text)
