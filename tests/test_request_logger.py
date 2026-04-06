"""
Unit tests for request_logger.py — _extract_session_id function and
per-session JSONL file writing behaviour of RequestLogger.

All tests are self-contained; no network calls or running services are needed.
"""

from __future__ import annotations

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from request_logger import RequestLogger, _extract_session_id


# ---------------------------------------------------------------------------
# _extract_session_id tests
# ---------------------------------------------------------------------------

class TestExtractSessionId:
    def test_valid_session_id_returned(self):
        """Returns the session_id when metadata.user_id is valid JSON with session_id."""
        body = {
            "metadata": {
                "user_id": json.dumps({"session_id": "abc-123"})
            }
        }
        assert _extract_session_id(body) == "abc-123"

    def test_valid_uuid_session_id(self):
        """A UUID-style session_id is returned unchanged (hyphens are allowed)."""
        session = "550e8400-e29b-41d4-a716-446655440000"
        body = {
            "metadata": {
                "user_id": json.dumps({"session_id": session})
            }
        }
        assert _extract_session_id(body) == session

    def test_session_id_with_underscores(self):
        """Underscores in session_id are allowed and returned as-is."""
        body = {
            "metadata": {
                "user_id": json.dumps({"session_id": "user_session_42"})
            }
        }
        assert _extract_session_id(body) == "user_session_42"

    def test_special_chars_sanitized(self):
        """Characters outside [a-zA-Z0-9_-] are replaced with underscores."""
        body = {
            "metadata": {
                "user_id": json.dumps({"session_id": "a/b\\c:d"})
            }
        }
        result = _extract_session_id(body)
        assert "/" not in result
        assert "\\" not in result
        assert ":" not in result
        # Original alphanumeric parts should be preserved
        assert "a" in result
        assert "b" in result

    def test_session_id_truncated_to_200_chars(self):
        """session_id values longer than 200 characters are truncated to 200."""
        long_id = "a" * 300
        body = {
            "metadata": {
                "user_id": json.dumps({"session_id": long_id})
            }
        }
        result = _extract_session_id(body)
        assert len(result) == 200
        assert result == "a" * 200

    def test_missing_metadata_returns_unknown(self):
        """Returns 'unknown' when metadata key is absent from the request body."""
        body = {"model": "claude-3", "messages": []}
        assert _extract_session_id(body) == "unknown"

    def test_empty_metadata_returns_unknown(self):
        """Returns 'unknown' when metadata is an empty dict."""
        body = {"metadata": {}}
        assert _extract_session_id(body) == "unknown"

    def test_missing_user_id_returns_unknown(self):
        """Returns 'unknown' when metadata exists but has no user_id key."""
        body = {"metadata": {"other_key": "value"}}
        assert _extract_session_id(body) == "unknown"

    def test_user_id_not_valid_json_returns_unknown(self):
        """Returns 'unknown' when user_id is not parseable as JSON."""
        body = {
            "metadata": {
                "user_id": "this is not json {{{"
            }
        }
        assert _extract_session_id(body) == "unknown"

    def test_user_id_json_missing_session_id_returns_unknown(self):
        """Returns 'unknown' when user_id JSON is valid but lacks session_id key."""
        body = {
            "metadata": {
                "user_id": json.dumps({"other_field": "value"})
            }
        }
        assert _extract_session_id(body) == "unknown"

    def test_user_id_session_id_empty_string_returns_unknown(self):
        """Returns 'unknown' when session_id in user_id JSON is an empty string."""
        body = {
            "metadata": {
                "user_id": json.dumps({"session_id": ""})
            }
        }
        assert _extract_session_id(body) == "unknown"

    def test_user_id_session_id_non_string_returns_unknown(self):
        """Returns 'unknown' when session_id in user_id JSON is not a string (e.g. int)."""
        body = {
            "metadata": {
                "user_id": json.dumps({"session_id": 12345})
            }
        }
        assert _extract_session_id(body) == "unknown"

    def test_user_id_session_id_null_returns_unknown(self):
        """Returns 'unknown' when session_id in user_id JSON is null/None."""
        body = {
            "metadata": {
                "user_id": json.dumps({"session_id": None})
            }
        }
        assert _extract_session_id(body) == "unknown"

    def test_empty_body_returns_unknown(self):
        """Returns 'unknown' for a completely empty dict body."""
        assert _extract_session_id({}) == "unknown"

    def test_user_id_is_empty_string_returns_unknown(self):
        """Returns 'unknown' when user_id is the empty string (json.loads('') raises)."""
        body = {
            "metadata": {
                "user_id": ""
            }
        }
        assert _extract_session_id(body) == "unknown"


# ---------------------------------------------------------------------------
# RequestLogger.__init__ / directory creation tests
# ---------------------------------------------------------------------------

class TestRequestLoggerInit:
    def test_creates_directory_if_not_exists(self, tmp_path):
        """RequestLogger creates the output directory when it does not exist."""
        new_dir = tmp_path / "new_logs"
        assert not new_dir.exists()
        rl = RequestLogger(str(new_dir))
        assert new_dir.is_dir()
        assert rl._dir == str(new_dir)

    def test_accepts_existing_directory(self, tmp_path):
        """RequestLogger succeeds when the output directory already exists."""
        rl = RequestLogger(str(tmp_path))
        assert rl._dir == str(tmp_path)

    def test_disabled_when_path_is_existing_file(self, tmp_path):
        """
        When output_dir points to an existing regular file, RequestLogger
        cannot create the directory, sets _dir to None and disables logging.
        """
        file_path = tmp_path / "not_a_dir"
        file_path.write_text("occupied")
        rl = RequestLogger(str(file_path))
        # os.makedirs raises OSError (EEXIST) — logger must disable itself
        assert rl._dir is None

    def test_disabled_when_path_not_writable(self, tmp_path):
        """When the directory exists but is not writable, logging is disabled."""
        locked_dir = tmp_path / "locked"
        locked_dir.mkdir()
        # Remove write permission
        locked_dir.chmod(0o555)
        try:
            rl = RequestLogger(str(locked_dir))
            assert rl._dir is None
        finally:
            # Restore permissions so tmp_path cleanup succeeds
            locked_dir.chmod(0o755)


# ---------------------------------------------------------------------------
# RequestLogger.log_entry — per-session file writing tests
# ---------------------------------------------------------------------------

class TestLogEntry:
    def test_writes_to_session_id_dot_jsonl(self, tmp_path):
        """log_entry writes to <output_dir>/<session_id>.jsonl."""
        rl = RequestLogger(str(tmp_path))
        rl.log_entry({"key": "value"}, session_id="sess-abc")
        expected = tmp_path / "sess-abc.jsonl"
        assert expected.exists(), f"Expected file {expected} was not created"

    def test_written_line_is_valid_json(self, tmp_path):
        """Each line written by log_entry is valid JSON."""
        rl = RequestLogger(str(tmp_path))
        entry = {"timestamp": "2026-01-01T00:00:00", "model": "test-model", "value": 42}
        rl.log_entry(entry, session_id="my-session")
        lines = (tmp_path / "my-session.jsonl").read_text(encoding="utf-8").splitlines()
        assert len(lines) == 1
        parsed = json.loads(lines[0])
        assert parsed == entry

    def test_multiple_entries_appended_as_separate_lines(self, tmp_path):
        """Calling log_entry multiple times appends each entry as a new line."""
        rl = RequestLogger(str(tmp_path))
        rl.log_entry({"n": 1}, session_id="session-x")
        rl.log_entry({"n": 2}, session_id="session-x")
        rl.log_entry({"n": 3}, session_id="session-x")
        lines = (tmp_path / "session-x.jsonl").read_text(encoding="utf-8").splitlines()
        assert len(lines) == 3
        assert [json.loads(l)["n"] for l in lines] == [1, 2, 3]

    def test_different_sessions_write_to_different_files(self, tmp_path):
        """Entries for different session_ids go to separate .jsonl files."""
        rl = RequestLogger(str(tmp_path))
        rl.log_entry({"session": "alpha"}, session_id="alpha")
        rl.log_entry({"session": "beta"}, session_id="beta")
        alpha_file = tmp_path / "alpha.jsonl"
        beta_file = tmp_path / "beta.jsonl"
        assert alpha_file.exists()
        assert beta_file.exists()
        alpha_data = json.loads(alpha_file.read_text())
        beta_data = json.loads(beta_file.read_text())
        assert alpha_data["session"] == "alpha"
        assert beta_data["session"] == "beta"

    def test_default_session_id_is_unknown(self, tmp_path):
        """When session_id is not specified, log_entry defaults to 'unknown.jsonl'."""
        rl = RequestLogger(str(tmp_path))
        rl.log_entry({"data": "test"})
        assert (tmp_path / "unknown.jsonl").exists()

    def test_no_write_when_dir_is_none(self, tmp_path):
        """log_entry is a no-op when _dir is None (logging disabled)."""
        rl = RequestLogger(str(tmp_path))
        rl._dir = None  # Simulate disabled state
        rl.log_entry({"data": "should_not_be_written"}, session_id="test")
        # No files should have been created
        assert list(tmp_path.iterdir()) == []


# ---------------------------------------------------------------------------
# RequestLogger.log_buffered_response — integration with _extract_session_id
# ---------------------------------------------------------------------------

class TestLogBufferedResponse:
    def _make_anthropic_response(self, text: str = "Hello!", input_tokens: int = 5,
                                  output_tokens: int = 3) -> bytes:
        payload = {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": text}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
        }
        return json.dumps(payload).encode()

    def test_writes_to_correct_session_file(self, tmp_path):
        """log_buffered_response routes to the file identified by session_id in metadata."""
        rl = RequestLogger(str(tmp_path))
        request_body = {
            "model": "claude-sonnet-4-6",
            "messages": [{"role": "user", "content": "hi"}],
            "metadata": {
                "user_id": json.dumps({"session_id": "session-42"})
            },
        }
        rl.log_buffered_response(
            request_body=request_body,
            model="claude-sonnet-4-6",
            endpoint="anthropic",
            response_body=self._make_anthropic_response("World"),
            timestamp="2026-01-01T00:00:00",
            status_code=200,
        )
        expected_file = tmp_path / "session-42.jsonl"
        assert expected_file.exists()
        entry = json.loads(expected_file.read_text())
        assert entry["model"] == "claude-sonnet-4-6"
        assert entry["response_text"] == "World"
        assert entry["input_tokens"] == 5
        assert entry["output_tokens"] == 3
        assert entry["stop_reason"] == "end_turn"
        assert entry["stream"] is False

    def test_falls_back_to_unknown_session_when_no_metadata(self, tmp_path):
        """When request body has no metadata, log_buffered_response writes to unknown.jsonl."""
        rl = RequestLogger(str(tmp_path))
        request_body = {
            "model": "claude-sonnet-4-6",
            "messages": [{"role": "user", "content": "hi"}],
        }
        rl.log_buffered_response(
            request_body=request_body,
            model="claude-sonnet-4-6",
            endpoint="anthropic",
            response_body=self._make_anthropic_response(),
            timestamp="2026-01-01T00:00:00",
        )
        assert (tmp_path / "unknown.jsonl").exists()

    def test_falls_back_to_unknown_session_when_user_id_invalid_json(self, tmp_path):
        """When user_id JSON is invalid, log_buffered_response writes to unknown.jsonl."""
        rl = RequestLogger(str(tmp_path))
        request_body = {
            "model": "claude-sonnet-4-6",
            "messages": [],
            "metadata": {"user_id": "NOT_VALID_JSON"},
        }
        rl.log_buffered_response(
            request_body=request_body,
            model="claude-sonnet-4-6",
            endpoint="anthropic",
            response_body=self._make_anthropic_response(),
            timestamp="2026-01-01T00:00:00",
        )
        assert (tmp_path / "unknown.jsonl").exists()
        # No other .jsonl files should have been created
        jsonl_files = list(tmp_path.glob("*.jsonl"))
        assert len(jsonl_files) == 1

    def test_entry_contains_request_body(self, tmp_path):
        """The logged entry includes the full request_body for later analysis."""
        rl = RequestLogger(str(tmp_path))
        request_body = {
            "model": "claude-sonnet-4-6",
            "messages": [{"role": "user", "content": "ping"}],
            "metadata": {"user_id": json.dumps({"session_id": "ping-session"})},
        }
        rl.log_buffered_response(
            request_body=request_body,
            model="claude-sonnet-4-6",
            endpoint="anthropic",
            response_body=self._make_anthropic_response("pong"),
            timestamp="2026-04-04T12:00:00",
        )
        entry = json.loads((tmp_path / "ping-session.jsonl").read_text())
        assert entry["request_body"] == request_body
        assert entry["timestamp"] == "2026-04-04T12:00:00"
        assert entry["endpoint"] == "anthropic"
