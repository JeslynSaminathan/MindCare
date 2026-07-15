"""
app.py

Flask application server for MindCare. Exposes:

  GET  /                      -> welcome.html (anonymous session gate)
  GET  /chat                  -> index.html   (chat + XAI dashboard)
  POST /api/session/start     -> creates an anonymous session, optional
                                  age_range/gender
  POST /api/chat              -> main chat turn: crisis check -> intent
                                  classification -> LIME explanation ->
                                  Qwen generation -> logged to SQLite
  GET  /api/history/<sid>     -> returns prior turns for a session
  GET  /api/health            -> liveness/readiness probe

The heavy models (DistilBERT classifier, Qwen 2.5 generator) are loaded
once at process startup via MindCareChatbot, not per-request.
"""

import os
import logging

from flask import Flask, request, jsonify, render_template, session

from chatbot import MindCareChatbot
from conversation_logger import ConversationLogger

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mindcare")

app = Flask(__name__)
app.secret_key = os.environ.get("MINDCARE_SECRET_KEY", "dev-secret-key-change-in-production")

# Set MINDCARE_SKIP_GENERATOR=1 for fast local dev / CI without downloading
# the multi-GB Qwen weights; falls back to canned intent responses.
_load_generator = os.environ.get("MINDCARE_SKIP_GENERATOR", "0") != "1"

logger.info("Initializing MindCare chatbot pipeline (this loads the ML models)...")
bot = MindCareChatbot(load_generator=_load_generator)
db = ConversationLogger(db_path=os.environ.get("MINDCARE_DB_PATH", "mindcare.db"))
logger.info("MindCare chatbot pipeline ready.")


# ---------------------------------------------------------------------------
# Page routes
# ---------------------------------------------------------------------------

@app.route("/")
def welcome():
    return render_template("welcome.html")


@app.route("/chat")
def chat_page():
    session_id = request.args.get("sid")
    if not session_id or not db.session_exists(session_id):
        return render_template("welcome.html")
    return render_template("index.html", session_id=session_id)


# ---------------------------------------------------------------------------
# API routes
# ---------------------------------------------------------------------------

@app.route("/api/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/api/session/start", methods=["POST"])
def start_session():
    payload = request.get_json(silent=True) or {}
    age_range = payload.get("age_range") or None
    gender = payload.get("gender") or None

    allowed_age_ranges = {None, "13-17", "18-24", "25-34", "35-44", "45+"}
    allowed_genders = {None, "female", "male", "nonbinary", "other"}

    if age_range not in allowed_age_ranges:
        return jsonify({"error": "invalid age_range"}), 400
    if gender not in allowed_genders:
        return jsonify({"error": "invalid gender"}), 400

    session_id = db.create_session(age_range=age_range, gender=gender)
    return jsonify({"session_id": session_id})


@app.route("/api/chat", methods=["POST"])
def chat():
    payload = request.get_json(silent=True) or {}
    session_id = payload.get("session_id")
    message = (payload.get("message") or "").strip()

    if not session_id or not db.session_exists(session_id):
        return jsonify({"error": "invalid or missing session_id"}), 400
    if not message:
        return jsonify({"error": "message must not be empty"}), 400
    if len(message) > 4000:
        return jsonify({"error": "message too long"}), 400

    history = db.get_history(session_id, limit=20)
    history_for_model = [{"role": h["role"], "content": h["content"]} for h in history]

    db.log_message(session_id, role="user", content=message)

    result = bot.handle_message(message, history=history_for_model, include_explanation=True)

    db.log_message(
        session_id,
        role="assistant",
        content=result["reply"],
        predicted_intent=result["intent"],
        intent_confidence=result["confidence"],
        crisis_tier=result["crisis_tier"],
    )

    if result["crisis_tier"] != "none":
        logger.warning(
            f"Crisis tier '{result['crisis_tier']}' detected for session {session_id}"
        )

    return jsonify(result)


@app.route("/api/history/<session_id>")
def history(session_id):
    if not db.session_exists(session_id):
        return jsonify({"error": "session not found"}), 404
    return jsonify({"history": db.get_history(session_id)})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug)
