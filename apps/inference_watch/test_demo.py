"""Web receipt, bounded trial, and crash recovery behavior."""

from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic, sleep
import unittest
from unittest.mock import patch
from uuid import uuid4

from fastapi.testclient import TestClient
import demo


class ReaderDemoTest(unittest.TestCase):
    def setUp(self):
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.enterContext(patch.object(demo, "DB", Path(tmp.name) / "demo.sqlite"))

    def ready(self, client, rid):
        end = monotonic() + 4
        while monotonic() < end:
            result = client.get("/api/requests/" + rid).json()
            if result["brief"]:
                return result
            sleep(0.01)
        self.fail("No completed brief")

    def test_scoped_three_receipt_trial(self):
        with TestClient(demo.app) as client:
            for sequence, domain in enumerate(
                ["code", "code", "math", "code", "code", "code", "code"]
            ):
                rid = str(uuid4())
                payload = {"id": rid, "workload": domain}
                for _ in range(2):
                    self.assertEqual(
                        client.post("/api/requests", json=payload).status_code, 200
                    )
                result = self.ready(client, rid)
                self.assertEqual(
                    result["variant"],
                    "comparison" if sequence in (3, 4, 5) else "summary",
                )
                path = "/api/requests/" + rid
                self.assertEqual(
                    client.post(
                        path + "/feedback", json={"choice": "undecided"}
                    ).status_code,
                    409,
                )
                for _ in range(2):
                    client.post(path + "/view")
                    response = client.post(
                        path + "/feedback", json={"choice": "undecided"}
                    )
                    self.assertEqual(response.status_code, 200)
                self.assertEqual(
                    client.post(
                        path + "/feedback", json={"choice": "draft"}
                    ).status_code,
                    409,
                )
            sums = client.get("/api/loop").json()
            self.assertEqual([r["workload"] for r in sums["trials"]], ["code"])
            self.assertEqual(sum(r["opened"] for r in sums["groups"]), 7)
            snapshot = client.get("/api/egbe-snapshot").json()
            self.assertEqual(
                sum(r.get("eventName") == "activated" for r in snapshot["rows"]), 7
            )
            self.assertNotIn(rid, str(snapshot))

    def test_pending_receipt_recovers_after_restart(self):
        rid = str(uuid4())
        with (
            patch.object(demo, "deliver", lambda: None),
            TestClient(demo.app) as client,
        ):
            client.post(
                "/api/requests", json={"id": rid, "workload": "chat"}
            ).raise_for_status()
            self.assertIsNone(client.get("/api/requests/" + rid).json()["brief"])
        with TestClient(demo.app) as client:
            self.assertIn("chat workloads", self.ready(client, rid)["brief"])
            group = client.get("/api/loop").json()["groups"][0]
            self.assertEqual((group["ready"], group["opened"]), (1, 0))


if __name__ == "__main__":
    unittest.main()
