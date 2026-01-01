"""Output utilities with timestamps."""

import sys
from datetime import datetime


def tprint(*args, **kwargs) -> None:
    """
    Print with a human-readable timestamp prefix.

    Works like print() but prepends [HH:MM:SS] to each line.
    Timestamps use local time.

    Args:
        *args: Arguments to print (same as print())
        **kwargs: Keyword arguments passed to print() (file, end, sep, flush)
    """
    timestamp = datetime.now().strftime("[%H:%M:%S]")

    # Handle the file kwarg (default to stdout)
    file = kwargs.get("file", sys.stdout)

    # Build the message from args
    sep = kwargs.get("sep", " ")
    message = sep.join(str(arg) for arg in args)

    # Prepend timestamp to each line
    lines = message.split("\n")
    timestamped_lines = [f"{timestamp} {line}" if line else line for line in lines]
    timestamped_message = "\n".join(timestamped_lines)

    # Print with remaining kwargs
    print_kwargs = {k: v for k, v in kwargs.items() if k in ("end", "flush", "file")}
    print(timestamped_message, **print_kwargs)
