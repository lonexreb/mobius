"""Exercise the CLI so every step, including retries, runs in a fresh process."""
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class WorkerTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.db = Path(directory.name) / "queue.sqlite"

    def call(self, *args, check=True):
        return subprocess.run(
            [sys.executable, str(Path(__file__).with_name("worker.py")), "--db", str(self.db), *args],
            text=True, capture_output=True, check=check,
        )

    def test_priority_and_recovery(self):
        self.call("request", "m1", "math")
        self.call("request", "c1", "code")
        self.call("request", "c2", "code")
        self.assertIn("2 code", self.call("tick").stdout)
        self.assertIn("Pending", self.call("brief", "m1", check=False).stderr)
        initial = self.call("brief", "c1").stdout
        self.assertIn("run_9fee8ebda7", initial)
        self.call("request", "c1", "code")
        self.assertIn("1 math", self.call("tick").stdout)
        self.assertEqual(self.call("tick").stdout.strip(), "idle")
        self.assertEqual(initial, self.call("brief", "c1").stdout)

    def test_conflicting_request_id(self):
        self.call("request", "a", "math")
        self.assertNotEqual(self.call("request", "a", "chat", check=False).returncode, 0)
        with sqlite3.connect(self.db) as db:
            self.assertEqual(db.execute("SELECT workload FROM requests").fetchall(), [("math",)])


if __name__ == "__main__":
    unittest.main()
