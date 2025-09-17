import re

# from boris.boriscore.db.db import DataBaseManager  # TODO: to be produced


class DataBaseManager:
    pass


def is_valid_email(email: str) -> bool:
    """
    Validates the email format using a regular expression.
    """
    pattern = r"^[\w\.-]+@[\w\.-]+\.\w+$"
    return re.match(pattern, email) is not None


def is_valid_password(password: str) -> bool:
    """
    Checks that the password is at least 8 characters long.
    """
    return len(password) >= 8


def is_unique_username_and_email(
    db_manager: DataBaseManager, username: str, email: str
) -> bool:
    """
    Checks the database to ensure the username or email does not already exist.
    """
    conn = db_manager._get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT COUNT(*) as count FROM accounts WHERE username = ? OR email = ?",
        (username, email),
    )
    result = cursor.fetchone()
    conn.close()
    return result["count"] == 0
