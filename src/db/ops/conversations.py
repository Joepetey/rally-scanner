"""Conversation history persistence — Claude agent state per Discord user."""

import json

from db.core.pool import get_conn


def get_conversation_history(
    discord_id: int,
    limit_messages: int = 10,
) -> list[dict]:
    """Get recent conversation history for a user.

    Returns list of message dicts in Claude API format.
    Truncates safely to avoid breaking tool_use/tool_result pairs.
    """
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT history FROM conversation_history WHERE discord_id = %s",
            (discord_id,),
        )
        row = cur.fetchone()

    if not row:
        return []

    history = json.loads(row["history"])
    history = history[-limit_messages:]

    def _is_tool_result(msg):
        content = msg.get("content")
        return isinstance(content, list) and any(
            isinstance(c, dict) and c.get("type") == "tool_result"
            for c in content
        )

    def _is_plain_user(msg):
        return msg.get("role") == "user" and not _is_tool_result(msg)

    for i, msg in enumerate(history):
        if _is_plain_user(msg):
            return history[i:]
    return []


def save_conversation_history(discord_id: int, history: list[dict]) -> None:
    """Save updated conversation history for a user."""
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO conversation_history (discord_id, updated_at, history) "
            "VALUES (%s, NOW(), %s) "
            "ON CONFLICT (discord_id) DO UPDATE SET "
            "history = EXCLUDED.history, updated_at = NOW()",
            (discord_id, json.dumps(history)),
        )


def clear_conversation_history(discord_id: int) -> None:
    """Clear conversation history for a user (start fresh conversation)."""
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "DELETE FROM conversation_history WHERE discord_id = %s",
            (discord_id,),
        )
