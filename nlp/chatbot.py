"""
Chatbot Inference and Response Engine.
Loads trained model pipeline, computes intent classification probabilities,
applies confidence thresholds, and retrieves conversational responses.
"""

import json
import os
import random
import sys
from datetime import datetime
from typing import Dict, Any, List, Optional

import joblib
import numpy as np

# Ensure module import paths
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from nlp.preprocessing import preprocess_text


CONFIDENCE_THRESHOLD = 0.25

SUGGESTED_QUESTIONS = [
    "What is Artificial Intelligence?",
    "Explain Natural Language Processing.",
    "What is machine learning?",
    "How does Python work?",
    "What are top final year project ideas?",
    "Tips for technical interview preparation",
    "Where can I learn AI and ML for free?",
    "Suggest some motivational words for coding"
]


class ChatbotEngine:
    """Chatbot inference engine managing model loading and response generation."""

    def __init__(self, model_path: str = "models/chatbot_model.pkl", dataset_path: str = "data/intents.json"):
        self.model_path = model_path
        self.dataset_path = dataset_path
        self.pipeline = None
        self.intents_data = None
        self.classes = []
        self.is_loaded = False
        
        self.load_model()

    def load_model(self) -> bool:
        """Load trained pipeline and intent response dictionary."""
        if not os.path.exists(self.model_path):
            print(f"[!] Warning: Model file {self.model_path} not found. Attempting to train automatically...")
            try:
                from nlp.train import train_and_evaluate
                train_and_evaluate(dataset_path=self.dataset_path, model_dir=os.path.dirname(self.model_path))
            except Exception as e:
                print(f"[!] Failed to auto-train model: {e}")
                return False

        try:
            artifact = joblib.load(self.model_path)
            self.pipeline = artifact['pipeline']
            self.intents_data = artifact['intents_data']
            self.classes = artifact['classes']
            self.is_loaded = True
            print(f"[✓] Chatbot model successfully loaded ({len(self.classes)} intents).")
            return True
        except Exception as e:
            print(f"[!] Error loading model artifact: {e}")
            self.is_loaded = False
            return False

    def predict_intent(self, text: str) -> Dict[str, Any]:
        """
        Classify intent for a given user query.
        Returns predicted intent, confidence score, and fallback indicator.
        """
        if not text or not text.strip():
            return {
                "intent": "fallback",
                "confidence": 0.0,
                "is_fallback": True,
                "reason": "empty_input"
            }

        if not self.is_loaded:
            if not self.load_model():
                return {
                    "intent": "fallback",
                    "confidence": 0.0,
                    "is_fallback": True,
                    "reason": "model_not_loaded"
                }

        cleaned = preprocess_text(text)
        if not cleaned.strip():
            return {
                "intent": "fallback",
                "confidence": 0.0,
                "is_fallback": True,
                "reason": "all_tokens_filtered"
            }

        try:
            # Predict probability distribution across classes
            probs = self.pipeline.predict_proba([cleaned])[0]
            max_idx = int(np.argmax(probs))
            confidence = float(probs[max_idx])
            predicted_tag = self.classes[max_idx]

            # Fallback condition: confidence below acceptable threshold
            if confidence < CONFIDENCE_THRESHOLD:
                return {
                    "intent": "fallback",
                    "detected_intent": predicted_tag,
                    "confidence": round(confidence, 4),
                    "is_fallback": True,
                    "reason": "low_confidence"
                }

            return {
                "intent": predicted_tag,
                "confidence": round(confidence, 4),
                "is_fallback": (predicted_tag == "fallback")
            }
        except Exception as e:
            return {
                "intent": "fallback",
                "confidence": 0.0,
                "is_fallback": True,
                "reason": f"prediction_exception: {str(e)}"
            }

    def get_response_for_tag(self, tag: str) -> str:
        """Fetch a random pre-configured response for an intent tag."""
        if not self.intents_data:
            return "I am here to help you learn about AI, NLP, and Python!"

        for intent in self.intents_data.get("intents", []):
            if intent.get("tag") == tag:
                responses = intent.get("responses", [])
                if responses:
                    return random.choice(responses)

        # Fallback responses if tag not found
        for intent in self.intents_data.get("intents", []):
            if intent.get("tag") == "fallback":
                responses = intent.get("responses", [])
                if responses:
                    return random.choice(responses)

        return "I'm sorry, I could not understand that. Could you please rephrase?"

    def get_response(self, user_text: str) -> Dict[str, Any]:
        """
        Process a user message and generate a structured response payload.
        """
        prediction = self.predict_intent(user_text)
        intent_tag = prediction["intent"]
        confidence = prediction["confidence"]
        is_fallback = prediction["is_fallback"]

        response_text = self.get_response_for_tag(intent_tag)
        
        timestamp = datetime.now().strftime("%I:%M %p")

        return {
            "query": user_text,
            "response": response_text,
            "intent": intent_tag,
            "confidence": confidence,
            "is_fallback": is_fallback,
            "timestamp": timestamp
        }

    def get_suggestions(self) -> List[str]:
        """Return curated list of suggested prompt questions."""
        return SUGGESTED_QUESTIONS


# Global singleton instance for application use
_bot_instance: Optional[ChatbotEngine] = None


def get_chatbot_instance() -> ChatbotEngine:
    """Return or initialize global ChatbotEngine singleton."""
    global _bot_instance
    if _bot_instance is None:
        _bot_instance = ChatbotEngine()
    return _bot_instance
