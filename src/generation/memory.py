"""
Conversation memory manager for DocAI.
Maintains multi-turn dialogue history, citation references, and query context.
"""

from typing import List, Dict, Any, Optional


class ConversationMemory:
    """Stores chat history and provides contextual slicing for follow-up questions."""

    def __init__(self):
        self.messages: List[Dict[str, Any]] = []

    def add_user_message(self, content: str) -> None:
        """Records a user query."""
        self.messages.append({
            "role": "user",
            "content": content
        })

    def add_assistant_message(
        self,
        content: str,
        citations: Optional[List[Dict[str, Any]]] = None,
        metrics: Optional[Dict[str, Any]] = None
    ) -> None:
        """Records an assistant response with associated citations and evaluation metrics."""
        self.messages.append({
            "role": "assistant",
            "content": content,
            "citations": citations or [],
            "metrics": metrics or {}
        })

    def get_recent_history(self, num_turns: int = 4) -> List[Dict[str, str]]:
        """Returns the last N dialogue turns formatted for query contextualization."""
        recent = self.messages[-num_turns:]
        return [{"role": m["role"], "content": m["content"]} for m in recent]

    def clear(self) -> None:
        """Resets the conversation history."""
        self.messages = []
