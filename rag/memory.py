"""
PART G: Innovation — Conversation Memory with Summarisation
Stores conversation turns; when history exceeds a threshold it summarises
older turns to preserve the gist without blowing up the context window.
This enables follow-up questions and multi-turn coherence.

Student: Maame Yaa Adumaba Appiah | Index: 10022200171
Course : CS4241 - Introduction to Artificial Intelligence | ACity 2026
"""
from __future__ import annotations
import json
import os

HISTORY_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs", "chat_history.json")
MAX_FULL_TURNS = 6   # keep last N turns verbatim
SUMMARY_EVERY = 10  # summarise after this many turns


class ConversationMemory:
    def __init__(self, session_id: str = "default"):
        self.session_id = session_id
        self.history: list[dict] = []  # [{"user": ..., "assistant": ...}]
        self.summary: str = ""

    def add_turn(self, user: str, assistant: str) -> None:
        self.history.append({"user": user, "assistant": assistant})
        # Keep only recent turns in memory
        if len(self.history) > MAX_FULL_TURNS:
            self.history = self.history[-MAX_FULL_TURNS:]

    def get_recent(self) -> list[dict]:
        return self.history[-MAX_FULL_TURNS:]

    def clear(self) -> None:
        self.history = []
        self.summary = ""

    def save(self) -> None:
        os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)
        data = {"session": self.session_id, "history": self.history, "summary": self.summary}
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def load(self) -> None:
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, encoding="utf-8") as f:
                data = json.load(f)
            if data.get("session") == self.session_id:
                self.history = data.get("history", [])
                self.summary = data.get("summary", "")

    def format_for_prompt(self) -> str:
        if not self.history:
            return ""
        lines = []
        if self.summary:
            lines.append(f"[Earlier conversation summary: {self.summary}]")
        for turn in self.get_recent():
            lines.append(f"User: {turn['user']}\nAssistant: {turn['assistant']}")
        return "\n\n".join(lines)
