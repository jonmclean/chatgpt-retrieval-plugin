import arrow

from typing import Optional

def to_unix_timestamp(date_str: str) -> int:
    """
    Convert a date string to a unix timestamp (seconds since epoch).

    Args:
        date_str: The date string to convert.

    Returns:
        The unix timestamp corresponding to the date string.

    If the date string cannot be parsed as a valid date format, returns the current unix timestamp and prints a warning.
    """
    # Try to parse the date string using arrow, which supports many common date formats
    try:
        date_obj = arrow.get(date_str)
        return int(date_obj.timestamp())
    except arrow.parser.ParserError:
        # If the parsing fails, return the current unix timestamp and print a warning
        print(f"Invalid date format: {date_str}")
        return int(arrow.now().timestamp())


def to_date_string(unix_timestamp: Optional[int]) -> str:
    """
    Convert a UNIX date into a string representation for return to the user
    Args:
        unix_timestamp: Unix timestamp of the date we want to format as string.

    Returns:
        The timestamp formatted as a string so it can be returned to the user in a JSON payload
    """
    if unix_timestamp is None:
        return None

    arrow_obj = arrow.get(unix_timestamp)
    return arrow_obj.format('YYYY-MM-DD HH:mm:ss')