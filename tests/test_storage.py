import sqlite3

from fastapi.testclient import TestClient

from fieldproof.api import create_app
from tests.support import ApiCase


V1_SCHEMA = """
CREATE TABLE trial (
    trial_id TEXT PRIMARY KEY, design TEXT NOT NULL, version INTEGER NOT NULL,
    phase TEXT NOT NULL CHECK (phase IN ('draft','open','closed')),
    close_cut INTEGER CHECK (close_cut >= 0));
CREATE TABLE site (
    trial_id TEXT NOT NULL REFERENCES trial(trial_id), site_id TEXT NOT NULL,
    data TEXT NOT NULL, PRIMARY KEY (trial_id, site_id));
CREATE TABLE enrollment (
    trial_id TEXT NOT NULL, unit_id TEXT NOT NULL, site_id TEXT NOT NULL,
    arm TEXT NOT NULL CHECK (arm IN ('standard','assisted')), data TEXT NOT NULL,
    PRIMARY KEY (trial_id, unit_id),
    FOREIGN KEY (trial_id, site_id) REFERENCES site(trial_id, site_id));
CREATE TABLE delivery (
    seq INTEGER PRIMARY KEY AUTOINCREMENT, trial_id TEXT NOT NULL REFERENCES trial(trial_id),
    delivery_id TEXT NOT NULL, digest TEXT NOT NULL, accepted INTEGER NOT NULL DEFAULT 0,
    repeated INTEGER NOT NULL DEFAULT 0, UNIQUE (trial_id, delivery_id));
CREATE TABLE signal (
    trial_id TEXT NOT NULL, event_id TEXT NOT NULL, revision INTEGER NOT NULL,
    unit_id TEXT NOT NULL, cycle_id TEXT NOT NULL,
    delivery_seq INTEGER NOT NULL REFERENCES delivery(seq), event TEXT NOT NULL,
    digest TEXT NOT NULL, PRIMARY KEY (trial_id, event_id, revision),
    FOREIGN KEY (trial_id, unit_id) REFERENCES enrollment(trial_id, unit_id));
CREATE INDEX signal_cut ON signal(trial_id, delivery_seq);
CREATE INDEX signal_cycle ON signal(trial_id, unit_id, cycle_id);
CREATE TABLE evidence (
    trial_id TEXT NOT NULL REFERENCES trial(trial_id), evidence_id TEXT NOT NULL,
    cut INTEGER NOT NULL, document TEXT NOT NULL,
    PRIMARY KEY (trial_id, evidence_id));
CREATE TABLE review (
    trial_id TEXT NOT NULL, evidence_id TEXT NOT NULL, market TEXT NOT NULL,
    desk TEXT NOT NULL CHECK (desk IN ('field','methods')),
    assessment TEXT NOT NULL CHECK (assessment IN ('endorse','object')),
    reviewer TEXT NOT NULL, note TEXT NOT NULL, recorded_at TEXT NOT NULL,
    PRIMARY KEY (trial_id, evidence_id, market, desk),
    FOREIGN KEY (trial_id, evidence_id) REFERENCES evidence(trial_id, evidence_id));
CREATE TABLE bulletin (
    trial_id TEXT NOT NULL, market TEXT NOT NULL, edition INTEGER NOT NULL,
    evidence_id TEXT NOT NULL,
    outcome TEXT NOT NULL CHECK (outcome IN ('advance','hold')),
    digest TEXT NOT NULL, document TEXT NOT NULL,
    PRIMARY KEY (trial_id, market, edition), UNIQUE (trial_id, market, evidence_id),
    FOREIGN KEY (trial_id, evidence_id) REFERENCES evidence(trial_id, evidence_id));
PRAGMA user_version=1;
"""


class StorageMigrationTest(ApiCase):
    def test_v1_database_migrates_and_keeps_parallel_flow_working(self):
        # Build a genuine v1 file in a separate path before the app opens it.
        legacy_path = self.db.parent / "legacy.sqlite3"
        legacy = sqlite3.connect(legacy_path)
        legacy.executescript(V1_SCHEMA)
        legacy.execute("INSERT INTO trial VALUES ('legacy','{}',1,'draft',NULL)")
        legacy.commit()
        legacy.close()

        client = TestClient(create_app(legacy_path))
        self.addCleanup(client.close)
        self.assertEqual(client.get("/health").status_code, 200)
        from fieldproof.storage import Store

        with Store(legacy_path).transaction() as db:
            self.assertEqual(db.execute("PRAGMA user_version").fetchone()[0], 2)
            columns = {row[1] for row in db.execute("PRAGMA table_info(trial)")}
            self.assertIn("close_schedule_seq", columns)
            tables = {
                row[0]
                for row in db.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            }
            self.assertIn("crossover_unit", tables)
            self.assertIn("crossover_schedule", tables)
            # The rebuilt signal table no longer forces a parallel enrollment FK.
            fk_tables = {row[2] for row in db.execute("PRAGMA foreign_key_list(signal)")}
            self.assertEqual(fk_tables, {"delivery"})
            preserved = db.execute(
                "SELECT trial_id, version, close_schedule_seq FROM trial"
            ).fetchone()
            self.assertEqual(tuple(preserved), ("legacy", 1, None))

    def test_fresh_database_is_created_at_version_two(self):
        path = self.db.parent / "fresh.sqlite3"
        from fieldproof.storage import Store

        Store(path)
        with sqlite3.connect(path) as db:
            self.assertEqual(db.execute("PRAGMA user_version").fetchone()[0], 2)
