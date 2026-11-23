"""交叉试验证据封存，复用与平行试验相同的证据/评审/公告表。"""
from __future__ import annotations

import csv
import io
import json
from sqlite3 import Connection

from fieldproof.analysis.evidence import select_window
from fieldproof.common import conflict, encoded
from fieldproof.crossover.analysis import build, current_schedule_seq
from fieldproof.design.registry import Trial, require_phase


def freeze(db: Connection, trial: Trial, expected_cut: int) -> dict:
    require_phase(trial, "closed")
    if trial.close_cut != expected_cut:
        raise conflict(
            "cut_changed",
            "The submitted cut does not match the closed trial",
            details={"close_cut": trial.close_cut},
        )
    expected_schedule = current_schedule_seq(db, trial.trial_id)
    document = build(
        db,
        trial,
        select_window(trial, None, None),
        expected_cut,
        expected_schedule,
    )
    document["close_schedule_seq"] = expected_schedule
    db.execute(
        """INSERT INTO evidence VALUES (?,?,?,?)
           ON CONFLICT(trial_id,evidence_id) DO NOTHING""",
        (trial.trial_id, document["evidence_id"], expected_cut, encoded(document)),
    )
    return document


def get(db: Connection, trial_id: str, evidence_id: str) -> dict:
    row = db.execute(
        "SELECT document FROM evidence WHERE trial_id=? AND evidence_id=?",
        (trial_id, evidence_id),
    ).fetchone()
    if row is None:
        from fieldproof.common import missing

        raise missing("evidence", evidence_id)
    return json.loads(row["document"])


def get_market(document: dict, market_id: str) -> dict:
    for item in document["market_results"]:
        if item["market_id"] == market_id:
            return item
    from fieldproof.common import missing

    raise missing("market", market_id)


def csv_metrics(document: dict) -> str:
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\r\n")
    writer.writerow((
        "evidence_id", "cut", "market_id", "cell", "sequence", "period", "arm",
        "metric", "unit", "value", "within_unit_effect",
    ))
    for market in document["market_results"]:
        for cell in market["cells"]:
            for sequence in cell["sequences"]:
                for period in sequence["periods"].values():
                    for metric, value in period["metrics"].items():
                        unit = "basis_points" if metric in ("manual", "alarm", "failure") else (
                            "Wh" if metric == "energy_wh" else "seconds"
                        )
                        writer.writerow((
                            document["evidence_id"], document["cut"], market["market_id"],
                            cell["cell"], sequence["sequence"], period["arm"], period["arm"],
                            metric, unit, value, "",
                        ))
                for metric, value in sequence["effects"].items():
                    unit = "basis_points" if metric in ("manual", "alarm", "failure") else (
                        "Wh" if metric == "energy_wh" else "seconds"
                    )
                    writer.writerow((
                        document["evidence_id"], document["cut"], market["market_id"],
                        cell["cell"], sequence["sequence"], "", "",
                        metric, unit, "", value,
                    ))
            for metric, value in cell["merged_effects"].items():
                unit = "basis_points" if metric in ("manual", "alarm", "failure") else (
                    "Wh" if metric == "energy_wh" else "seconds"
                )
                writer.writerow((
                    document["evidence_id"], document["cut"], market["market_id"],
                    cell["cell"], "merged", "", "", metric, unit, "", value,
                ))
    return output.getvalue()
