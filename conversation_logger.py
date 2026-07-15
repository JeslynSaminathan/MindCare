"""
conversation_logger.py

SQLite adapter for anonymous conversation telemetry logging.

Privacy design notes:
- No names, emails, or freeform account identifiers are ever stored.
- Each session is identified only by a randomly generated session_id (UUID4)
  minted client-side / at session start -- it cannot be reversed to a real
  identity.
- Optional demographic fields (age_range, gender) are stored only if the
  user explicitly opted in on the welcome screen; both default to NULL.
- Message text is stored to support the "improve MindCare" telemetry the
  welcome screen discloses, but no other user-identifying metadata (IP,
  device fingerprint, etc.) is captured by this module.
"""

import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any

DB_PATH = "mindcare.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    age_range TEXT,
    gender TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    message_id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
    content TEXT NOT NULL,
    predicted_intent TEXT,
    intent_confidence REAL,
    crisis_tier TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY(session_id) REFERENCES sessions(session_id)
);

CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ConversationLogger:
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._init_db()

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(SCHEMA)

    # -- Session management -------------------------------------------------

    def create_session(self, age_range: Optional[str] = None, gender: Optional[str] = None) -> str:
        """Create a new anonymous session and return its session_id."""
        session_id = str(uuid.uuid4())
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO sessions (session_id, age_range, gender, created_at) VALUES (?, ?, ?, ?)",
                (session_id, age_range or None, gender or None, _now()),
            )
        return session_id

    def session_exists(self, session_id: str) -> bool:
        with self._connect() as conn:
            cur = conn.execute("SELECT 1 FROM sessions WHERE session_id = ?", (session_id,))
            return cur.fetchone() is not None

    # -- Message logging ------------------------------------------------------

    def log_message(
        self,
        session_id: str,
        role: str,
        content: str,
        predicted_intent: Optional[str] = None,
        intent_confidence: Optional[float] = None,
        crisis_tier: Optional[str] = None,
    ) -> int:
        if role not in ("user", "assistant"):
            raise ValueError("role must be 'user' or 'assistant'")

        with self._connect() as conn:
            cur = conn.execute(
                """INSERT INTO messages
                   (session_id, role, content, predicted_intent, intent_confidence, crisis_tier, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (session_id, role, content, predicted_intent, intent_confidence, crisis_tier, _now()),
            )
            return cur.lastrowid

    def get_history(self, session_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            cur = conn.execute(
                """SELECT role, content, predicted_intent, intent_confidence, crisis_tier, created_at
                   FROM messages WHERE session_id = ? ORDER BY message_id ASC LIMIT ?""",
                (session_id, limit),
            )
            rows = cur.fetchall()

        return [
            {
                "role": r[0],
                "content": r[1],
                "predicted_intent": r[2],
                "intent_confidence": r[3],
                "crisis_tier": r[4],
                "created_at": r[5],
            }
            for r in rows
        ]
