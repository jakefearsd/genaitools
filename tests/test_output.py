"""Tests for genaitools/output.py"""

import re
import sys
from io import StringIO

from genaitools.output import tprint


class TestTprint:
    """Tests for the tprint function."""

    def test_tprint_has_timestamp(self):
        """Verify tprint outputs with timestamp prefix."""
        output = StringIO()
        tprint("test message", file=output)
        result = output.getvalue()
        # Should match [HH:MM:SS] pattern
        assert re.match(r"\[\d{2}:\d{2}:\d{2}\] test message\n", result)

    def test_tprint_multiline(self):
        """Verify tprint timestamps each line."""
        output = StringIO()
        tprint("line1\nline2\nline3", file=output)
        result = output.getvalue()
        lines = result.strip().split("\n")
        assert len(lines) == 3
        for line in lines:
            assert re.match(r"\[\d{2}:\d{2}:\d{2}\] line\d", line)

    def test_tprint_empty_lines_preserved(self):
        """Verify empty lines in multiline output are preserved without timestamp."""
        output = StringIO()
        tprint("line1\n\nline3", file=output)
        result = output.getvalue()
        lines = result.split("\n")
        # Should be: "[HH:MM:SS] line1", "", "[HH:MM:SS] line3", ""
        assert len(lines) == 4
        assert lines[1] == ""  # Empty line preserved

    def test_tprint_stderr(self):
        """Verify tprint works with stderr."""
        output = StringIO()
        tprint("error message", file=output)
        result = output.getvalue()
        assert "error message" in result

    def test_tprint_multiple_args(self):
        """Verify tprint handles multiple arguments like print()."""
        output = StringIO()
        tprint("hello", "world", file=output)
        result = output.getvalue()
        assert "hello world" in result

    def test_tprint_custom_sep(self):
        """Verify tprint respects sep argument."""
        output = StringIO()
        tprint("a", "b", "c", sep="-", file=output)
        result = output.getvalue()
        assert "a-b-c" in result
