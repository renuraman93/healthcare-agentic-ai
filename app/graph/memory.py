"""
Short-term conversation memory, keyed by session id.

In-memory only: nothing is written to disk, and it is lost on restart.
That is deliberate for a healthcare demo, since it avoids persisting
conversations that might contain sensitive text.
"""

from collections import deque


class ConversationMemory:
    def __init__(self, max_messages: int = 10) -> None:
        self._store: dict[str, deque] = {}
        self._max_messages = max_messages

    def get(self, session_id: str) -> list[dict]:
        return list(self._store.get(session_id, []))

    def add(self, session_id: str, role: str, content: str) -> None:
        self._store.setdefault(session_id, deque(maxlen=self._max_messages)).append(
            {"role": role, "content": content}
        )

    def clear(self, session_id: str) -> None:
        self._store.pop(session_id, None)