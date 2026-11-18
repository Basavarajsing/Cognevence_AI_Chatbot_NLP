"""
AI Chatbot Application - Flask Web Backend.
Implements chat API endpoints, session history management, real-time analytics,
conversation export features, and dashboard rendering.
"""

import json
import os
import threading
from datetime import datetime
from flask import (
    Flask,
    render_template,
    request,
    jsonify,
    session,
    Response,
    send_file,
    send_from_directory
)

from nlp.chatbot import get_chatbot_instance

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "academic-nlp-chatbot-secret-key-2026")
app.config['JSON_SORT_KEYS'] = False

ANALYTICS_FILE = os.path.join("data", "analytics.json")
analytics_lock = threading.Lock()


def load_persistent_analytics() -> dict:
    """Load real runtime analytics from disk or initialize default state."""
    if os.path.exists(ANALYTICS_FILE):
        try:
            with open(ANALYTICS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass

    return {
        "total_messages": 0,
        "total_queries": 0,
        "fallback_count": 0,
        "confidence_sum": 0.0,
        "intents_count": {},
        "recent_queries": []
    }


def save_persistent_analytics(data: dict):
    """Save updated runtime analytics to disk safely."""
    try:
        os.makedirs(os.path.dirname(ANALYTICS_FILE), exist_ok=True)
        with open(ANALYTICS_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
    except Exception as e:
        app.logger.error(f"Error saving analytics: {e}")


def update_analytics(query: str, intent: str, confidence: float, is_fallback: bool):
    """Thread-safe update of live application analytics."""
    with analytics_lock:
        analytics = load_persistent_analytics()
        analytics["total_messages"] += 2  # user message + bot response
        analytics["total_queries"] += 1
        analytics["confidence_sum"] += confidence

        if is_fallback:
            analytics["fallback_count"] += 1

        # Track intent distribution
        analytics["intents_count"][intent] = analytics["intents_count"].get(intent, 0) + 1

        # Track last 20 queries
        recent = analytics.get("recent_queries", [])
        recent.insert(0, {
            "query": query[:60] + ("..." if len(query) > 60 else ""),
            "intent": intent,
            "confidence": round(confidence, 4),
            "is_fallback": is_fallback,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        analytics["recent_queries"] = recent[:20]

        save_persistent_analytics(analytics)


# Preload chatbot engine
chatbot = get_chatbot_instance()


@app.route("/")
def index():
    """Render main chatbot conversation interface."""
    if "history" not in session:
        session["history"] = []
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    """
    Core Chat API Endpoint:
    Processes user query, performs intent classification,
    updates chat history and analytics, and returns bot response payload.
    """
    data = request.get_json(silent=True)
    if not data or "message" not in data:
        return jsonify({
            "error": "Invalid request payload. 'message' field is required."
        }), 400

    raw_message = data["message"]
    if not isinstance(raw_message, str):
        return jsonify({"error": "Message must be a string."}), 400

    user_message = raw_message.strip()
    if not user_message:
        return jsonify({
            "error": "Empty message.",
            "response": "Please type a question or select one of the suggested topics!",
            "intent": "fallback",
            "confidence": 0.0,
            "is_fallback": True,
            "timestamp": datetime.now().strftime("%I:%M %p")
        }), 200

    if len(user_message) > 1000:
        return jsonify({
            "error": "Message too long. Please restrict query to 1000 characters."
        }), 400

    # Process query through NLP engine
    bot_result = chatbot.get_response(user_message)

    # Initialize or update session history
    if "history" not in session:
        session["history"] = []

    history = session["history"]
    history.append({
        "role": "user",
        "message": user_message,
        "timestamp": datetime.now().strftime("%I:%M %p")
    })
    history.append({
        "role": "bot",
        "message": bot_result["response"],
        "intent": bot_result["intent"],
        "confidence": bot_result["confidence"],
        "is_fallback": bot_result["is_fallback"],
        "timestamp": bot_result["timestamp"]
    })
    session["history"] = history
    session.modified = True

    # Update real-time analytics
    update_analytics(
        query=user_message,
        intent=bot_result["intent"],
        confidence=bot_result["confidence"],
        is_fallback=bot_result["is_fallback"]
    )

    return jsonify({
        "response": bot_result["response"],
        "intent": bot_result["intent"],
        "confidence": bot_result["confidence"],
        "is_fallback": bot_result["is_fallback"],
        "timestamp": bot_result["timestamp"]
    }), 200


@app.route("/suggested", methods=["GET"])
def suggested():
    """Feature A: Return smart suggested starter questions."""
    return jsonify({
        "suggestions": chatbot.get_suggestions()
    }), 200


@app.route("/clear", methods=["POST"])
def clear_history():
    """Clear session chat history."""
    session["history"] = []
    session.modified = True
    return jsonify({
        "status": "success",
        "message": "Conversation history successfully cleared."
    }), 200


@app.route("/results/<path:filename>")
def serve_results(filename):
    """Serve evaluation artifacts like confusion_matrix.png."""
    return send_from_directory("results", filename)


@app.route("/dashboard")
def dashboard():
    """Feature B: Render Chat Analytics Dashboard."""
    return render_template("dashboard.html")


@app.route("/api/analytics", methods=["GET"])
def get_analytics():
    """
    Feature B API: Return live computed analytics data
    and model training evaluation metrics.
    """
    analytics = load_persistent_analytics()
    total_queries = analytics.get("total_queries", 0)
    total_messages = analytics.get("total_messages", 0)
    fallback_count = analytics.get("fallback_count", 0)
    confidence_sum = analytics.get("confidence_sum", 0.0)
    intents_count = analytics.get("intents_count", {})

    avg_confidence = round((confidence_sum / total_queries), 4) if total_queries > 0 else 0.0
    fallback_rate = round((fallback_count / total_queries * 100), 2) if total_queries > 0 else 0.0

    # Sort intents by frequency
    sorted_intents = sorted(intents_count.items(), key=lambda x: x[1], reverse=True)
    top_intents = [{"intent": k, "count": v} for k, v in sorted_intents[:10]]

    # Load offline model metrics from results
    metrics_path = os.path.join("results", "metrics.json")
    model_metrics = {}
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, "r", encoding="utf-8") as f:
                model_metrics = json.load(f)
        except Exception:
            pass

    return jsonify({
        "total_messages": total_messages,
        "total_queries": total_queries,
        "fallback_count": fallback_count,
        "fallback_rate": fallback_rate,
        "avg_confidence": avg_confidence,
        "top_intents": top_intents,
        "all_intents_distribution": intents_count,
        "recent_queries": analytics.get("recent_queries", []),
        "model_metrics": model_metrics
    }), 200


@app.route("/export/json", methods=["GET"])
def export_json():
    """Feature C: Export conversation history as a JSON download."""
    history = session.get("history", [])
    export_payload = {
        "export_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_messages": len(history),
        "application": "AI Chatbot Using Natural Language Processing",
        "conversation": history
    }
    
    filename = f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    return Response(
        json.dumps(export_payload, indent=4),
        mimetype="application/json",
        headers={"Content-Disposition": f"attachment;filename={filename}"}
    )


@app.route("/export/txt", methods=["GET"])
def export_txt():
    """Feature C: Export conversation history as a formatted text transcript."""
    history = session.get("history", [])
    lines = [
        "=" * 65,
        "        AI CHATBOT CONVERSATION TRANSCRIPT",
        f"        Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 65,
        ""
    ]

    if not history:
        lines.append("No conversation history recorded in this session.\n")
    else:
        for idx, item in enumerate(history, 1):
            role = "USER" if item.get("role") == "user" else "AI CHATBOT"
            timestamp = item.get("timestamp", "")
            message = item.get("message", "")
            lines.append(f"[{idx}] {role} ({timestamp}):")
            lines.append(f"    {message}")
            if item.get("role") == "bot":
                intent = item.get("intent", "N/A")
                conf = item.get("confidence", 0.0)
                lines.append(f"    [Detected Intent: {intent} | Confidence: {conf * 100:.1f}%]")
            lines.append("-" * 65)

    filename = f"chat_transcript_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    return Response(
        "\n".join(lines),
        mimetype="text/plain",
        headers={"Content-Disposition": f"attachment;filename={filename}"}
    )


@app.route("/health", methods=["GET"])
def health():
    """Service health check endpoint."""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": chatbot.is_loaded,
        "classes_count": len(chatbot.classes)
    }), 200


@app.errorhandler(404)
def not_found(e):
    if request.path.startswith("/api/"):
        return jsonify({"error": "Endpoint not found"}), 404
    return render_template("index.html"), 404


@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "Internal server error occurred."}), 500


if __name__ == "__main__":
    print("[*] Starting AI Chatbot Flask Server on http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False)
