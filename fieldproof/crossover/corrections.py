"""日程或执行证据更正后，判定受影响的期与配对。"""
from __future__ import annotations

from sqlite3 import Connection

from fieldproof.analysis.cohorts import AnalysisWindow
from fieldproof.crossover.analysis import build, load
from fieldproof.design.registry import Trial


def _index(material) -> dict[tuple[str, str], object]:
    return {(item.unit_id, item.cycle_id): item for item in material.classified}


def _pairing_changes(before_rows: dict, after_rows: dict, trial: Trial) -> list[dict]:
    changes = []
    for unit_id in sorted(before_rows.keys() | after_rows.keys()):
        old = before_rows.get(unit_id)
        new = after_rows.get(unit_id)
        old_status = old.pairing(trial) if old else None
        new_status = new.pairing(trial) if new else None
        if old_status == new_status:
            continue
        changes.append({
            "unit_id": unit_id,
            "before_pairing": old_status,
            "after_pairing": new_status,
            "sequence": new.unit.sequence if new else old.unit.sequence,
            "cell": new.cell if new else old.cell,
        })
    return changes


def _cycle_changes(before_material, after_material) -> list[dict]:
    before = _index(before_material)
    after = _index(after_material)
    changes = []
    for key in sorted(before.keys() | after.keys()):
        old = before.get(key)
        new = after.get(key)
        old_payload = old.payload() if old else None
        new_payload = new.payload() if new else None
        if old_payload == new_payload:
            continue
        unit_id, cycle_id = key
        changes.append({
            "unit_id": unit_id,
            "cycle_id": cycle_id,
            "before": None if old is None else {
                "nominal_period": old.nominal_period,
                "accepted_period": old.accepted_period,
                "outside_reason": old.outside_reason,
                "state": old.cycle.state,
                "source_revisions": old.cycle.sources,
            },
            "after": None if new is None else {
                "nominal_period": new.nominal_period,
                "accepted_period": new.accepted_period,
                "outside_reason": new.outside_reason,
                "state": new.cycle.state,
                "source_revisions": new.cycle.sources,
            },
            "became_suspect": bool(
                old and old.accepted_period is not None and new and new.accepted_period is None
            ),
            "affected_period": (
                old.accepted_period if old and old.accepted_period is not None else None
            ),
        })
    return changes


def impact(
    db: Connection,
    trial: Trial,
    window: AnalysisWindow,
    before_cut: int,
    after_cut: int,
    before_schedule_seq: int,
    after_schedule_seq: int,
) -> dict:
    if before_cut > after_cut:
        from fieldproof.common import invalid

        raise invalid("cut_order", "The earlier cut must be no later than the later cut")
    if before_schedule_seq > after_schedule_seq:
        from fieldproof.common import invalid

        raise invalid("schedule_order", "The earlier schedule revision must be no later")
    before = load(db, trial, window, before_cut, before_schedule_seq)
    after = load(db, trial, window, after_cut, after_schedule_seq)
    cycle_changes = _cycle_changes(before, after)
    pairing_changes = _pairing_changes(before.rows, after.rows, trial)

    affected_pairs = []
    for change in cycle_changes:
        if not change["became_suspect"]:
            continue
        row = after.rows.get(change["unit_id"])
        if row is None:
            continue
        affected_pairs.append({
            "unit_id": change["unit_id"],
            "cycle_id": change["cycle_id"],
            "sequence": row.unit.sequence,
            "cell": row.cell,
            "affected_period": change["affected_period"],
            "affected_arm": row.period_arm(change["affected_period"]),
            "outside_reason": change["after"]["outside_reason"],
            "pairing_after": row.pairing(trial),
        })

    before_doc = build(db, trial, window, before_cut, before_schedule_seq)
    after_doc = build(db, trial, window, after_cut, after_schedule_seq)
    market_verdicts = [
        {
            "market_id": new["market_id"],
            "before_recommendation": old["recommendation"],
            "after_recommendation": new["recommendation"],
            "before_blocked_by": old["blocked_by"],
            "after_blocked_by": new["blocked_by"],
        }
        for old, new in zip(before_doc["market_results"], after_doc["market_results"], strict=True)
    ]
    return {
        "window": window.payload(),
        "before_cut": before_cut,
        "after_cut": after_cut,
        "before_schedule_seq": before_schedule_seq,
        "after_schedule_seq": after_schedule_seq,
        "total_changed_cycles": len(cycle_changes),
        "suspect_cross_period_cycles": affected_pairs,
        "pairing_changes": pairing_changes,
        "markets": market_verdicts,
        "cycles": cycle_changes,
    }
