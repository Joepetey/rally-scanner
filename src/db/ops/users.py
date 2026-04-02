"""User account persistence — Discord user registration and capital tracking."""

from db.core.pool import get_conn


def ensure_user(discord_id: int, username: str) -> None:
    """Insert or update a user. Called at the start of every command."""
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO users (discord_id, username) VALUES (%s, %s) "
            "ON CONFLICT (discord_id) DO UPDATE SET username = EXCLUDED.username",
            (discord_id, username),
        )


def set_capital(discord_id: int, capital: float) -> None:
    """Set the user's portfolio capital."""
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "UPDATE users SET capital = %s WHERE discord_id = %s",
            (capital, discord_id),
        )


def get_capital(discord_id: int) -> float:
    """Get the user's portfolio capital. Returns 0.0 if not set."""
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT capital FROM users WHERE discord_id = %s", (discord_id,)
        )
        row = cur.fetchone()
    return float(row["capital"]) if row and row["capital"] else 0.0
