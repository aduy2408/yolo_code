from __future__ import annotations

import json
import stat
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from utils.marimo_client import (
    MarimoClient,
    MarimoClientError,
    MarimoConfig,
    load_config,
    save_config,
)


class MarimoClientTests(unittest.TestCase):
    def test_environment_config_wins_over_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(json.dumps({"url": "https://file", "token": "file-token"}))
            config = load_config(path, {"MARIMO_URL": "https://env/", "MARIMO_TOKEN": "env-token"})
            self.assertEqual(config, MarimoConfig("https://env", "env-token"))

    def test_save_config_is_owner_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nested" / "config.json"
            save_config("https://server/", "token", path)
            self.assertEqual(json.loads(path.read_text())["url"], "https://server")
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)

    def test_missing_config_is_actionable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(MarimoClientError):
                MarimoClient(load_config(Path(tmp) / "missing.json"))

    def test_execute_retries_stale_session_and_yields_events(self) -> None:
        session = Mock()
        sessions_response = Mock(status_code=200)
        sessions_response.json.return_value = {"stale": {}, "active": {}}
        sessions_response.raise_for_status.return_value = None
        stale_response = Mock(status_code=410)
        active_response = Mock(status_code=200)
        active_response.iter_lines.return_value = iter([
            "event: stdout",
            'data: {"data": "hello"}',
            "event: done",
            'data: {"success": true}',
        ])
        session.get.return_value = sessions_response
        session.post.side_effect = [stale_response, active_response]

        client = MarimoClient(MarimoConfig("https://server", "token"), session=session)
        events = client.execute("print('hello')")
        self.assertEqual(events, [("stdout", {"data": "hello"}), ("done", {"success": True})])
        self.assertEqual(session.post.call_count, 2)


if __name__ == "__main__":
    unittest.main()
