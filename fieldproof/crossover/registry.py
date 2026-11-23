"""两期交叉试验的登记、本地日程与开启前核对。"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from sqlite3 import Connection

from fieldproof.common import conflict, encoded, invalid, missing, utc_now
from fieldproof.design.contracts import (
    CrossoverEnrollment,
    Enrollment,
    ScheduleCorrection,
    Site,
    TrialPlan,
)
from fieldproof.design.registry import Roster, Trial, check_version, require_phase
from fieldproof.signals.ledger import current_cut

SEQUENCES = ("ab", "ba")


@dataclass(frozen=True)
class UnitSchedule:
    first_from: date
    first_until: date
    second_from: date
    second_until: date

    @classmethod
    def from_payload(cls, data: dict) -> "UnitSchedule":
        return cls(
            date.fromisoformat(data["first_from"]),
            date.fromisoformat(data["first_until"]),
            date.fromisoformat(data["second_from"]),
            date.fromisoformat(data["second_until"]),
        )

    def payload(self) -> dict:
        return {
            "first_from": self.first_from.isoformat(),
            "first_until": self.first_until.isoformat(),
            "second_from": self.second_from.isoformat(),
            "second_until": self.second_until.isoformat(),
        }

    def bounds(self, period: int) -> tuple[date, date]:
        return (self.first_from, self.first_until) if period == 1 else (
            self.second_from,
            self.second_until,
        )

    def washout_days(self) -> int:
        return (self.second_from - self.first_until).days

    def period_starting_on(self, day: date) -> int | None:
        """仅凭本地开始日期的归属（用于判定名义分配，不代表有效配对）。"""
        if self.first_from <= day < self.first_until:
            return 1
        if self.second_from <= day < self.second_until:
            return 2
        return None


@dataclass(frozen=True)
class CrossoverUnit:
    unit_id: str
    site_id: str
    sequence: str
    load: str
    revision: int
    schedule: UnitSchedule

    def arm_in(self, period: int) -> str:
        if self.sequence == "ab":
            return "standard" if period == 1 else "assisted"
        return "assisted" if period == 1 else "standard"

    def payload(self) -> dict:
        return {
            "unit_id": self.unit_id,
            "site_id": self.site_id,
            "sequence": self.sequence,
            "load": self.load,
            "schedule_revision": self.revision,
            **self.schedule.payload(),
        }


def require_crossover(trial: Trial):
    if not trial.plan.is_crossover:
        raise conflict("not_crossover", "This trial uses parallel allocation")


def require_parallel(trial: Trial):
    if trial.plan.is_crossover:
        raise conflict("not_parallel", "This trial uses two-period crossover allocation")


def _load_rows(db: Connection, trial_id: str) -> list:
    return db.execute(
        "SELECT data, schedule_seq FROM crossover_unit WHERE trial_id=? ORDER BY unit_id",
        (trial_id,),
    ).fetchall()


def load_units(db: Connection, trial_id: str) -> dict[str, CrossoverUnit]:
    units = {}
    for row in _load_rows(db, trial_id):
        data = json.loads(row["data"])
        schedule_row = db.execute(
            "SELECT data FROM crossover_schedule WHERE seq=?", (row["schedule_seq"],)
        ).fetchone()
        schedule = UnitSchedule.from_payload(json.loads(schedule_row["data"]))
        units[data["unit_id"]] = CrossoverUnit(
            data["unit_id"], data["site_id"], data["sequence"], data["load"],
            data["schedule_revision"], schedule,
        )
    return units


def _synthetic_enrollment(unit: CrossoverUnit) -> Enrollment:
    return Enrollment(
        unit_id=unit.unit_id,
        site_id=unit.site_id,
        arm=unit.arm_in(1),
        load=unit.load,
        active_from=unit.schedule.first_from,
        active_until=unit.schedule.second_until,
    )


def crossover_roster(db: Connection, trial: Trial) -> Roster:
    """把交叉名册适配成周期解析所需的站点/设备视图。"""
    sites = {
        row["site_id"]: Site.model_validate_json(row["data"])
        for row in db.execute(
            "SELECT site_id, data FROM site WHERE trial_id=? ORDER BY site_id",
            (trial.trial_id,),
        )
    }
    units = {
        unit.unit_id: _synthetic_enrollment(unit)
        for unit in load_units(db, trial.trial_id).values()
    }
    return Roster(sites, units)


def validate_crossover_unit(plan: TrialPlan, sites: dict[str, Site], unit: CrossoverEnrollment):
    site = sites.get(unit.site_id)
    if site is None:
        raise missing("site", unit.site_id)
    market = plan.market(site.market_id)
    if market is None or not any(
        cell.climate == site.climate and cell.load == unit.load for cell in market.cells
    ):
        raise invalid("unplanned_cell", f"Unit {unit.unit_id} has no planned cell")
    schedule = UnitSchedule(
        unit.first_from, unit.first_until, unit.second_from, unit.second_until
    )
    if schedule.first_from < plan.starts_on or schedule.second_until > plan.ends_before:
        raise invalid("enrollment_window", f"Unit {unit.unit_id} falls outside the trial")
    washout = plan.crossover.policy.minimum_washout_days
    if schedule.washout_days() < washout:
        raise invalid(
            "washout_too_short",
            f"Unit {unit.unit_id} needs at least {washout} washout dates",
            details={"observed": schedule.washout_days(), "required": washout},
        )


def enroll_crossover(db: Connection, trial: Trial, units: list[CrossoverEnrollment]) -> dict:
    require_phase(trial, "draft")
    sites = {
        row["site_id"]: Site.model_validate_json(row["data"])
        for row in db.execute(
            "SELECT site_id, data FROM site WHERE trial_id=?", (trial.trial_id,)
        )
    }
    current = load_units(db, trial.trial_id)
    new = []
    for unit in units:
        validate_crossover_unit(trial.plan, sites, unit)
        previous = current.get(unit.unit_id)
        if previous is None:
            new.append(unit)
        elif (
            previous.sequence != unit.sequence
            or previous.load != unit.load
            or previous.site_id != unit.site_id
        ):
            raise conflict(
                "unit_exists", f"Unit {unit.unit_id} has a different crossover assignment"
            )
    if not new:
        return {"added": 0, "already_enrolled": len(units), "version": trial.version}
    for unit in new:
        result = db.execute(
            """INSERT INTO crossover_schedule
               (trial_id, unit_id, revision, reason, recorded_at, recorded_cut, data)
               VALUES (?,?,?,?,?,?,?)""",
            (
                trial.trial_id, unit.unit_id, 1, "initial_registration",
                utc_now(), 0, encoded(unit.schedule_payload()),
            ),
        )
        document = {
            "unit_id": unit.unit_id,
            "site_id": unit.site_id,
            "sequence": unit.sequence,
            "load": unit.load,
            "schedule_revision": 1,
        }
        db.execute(
            "INSERT INTO crossover_unit VALUES (?,?,?,?,?,?)",
            (
                trial.trial_id, unit.unit_id, unit.site_id, unit.sequence,
                result.lastrowid, encoded(document),
            ),
        )
    db.execute("UPDATE trial SET version=version+1 WHERE trial_id=?", (trial.trial_id,))
    return {
        "added": len(new),
        "already_enrolled": len(units) - len(new),
        "version": _version(db, trial.trial_id),
    }


def _version(db: Connection, trial_id: str) -> int:
    return db.execute(
        "SELECT version FROM trial WHERE trial_id=?", (trial_id,)
    ).fetchone()["version"]


def schedule_history(db: Connection, trial: Trial, unit_id: str) -> list[dict]:
    if unit_id not in load_units(db, trial.trial_id):
        raise missing("crossover_unit", unit_id)
    rows = db.execute(
        """SELECT revision, reason, recorded_at, recorded_cut, data FROM crossover_schedule
           WHERE trial_id=? AND unit_id=? ORDER BY revision""",
        (trial.trial_id, unit_id),
    )
    return [
        {
            "revision": row["revision"],
            "reason": row["reason"],
            "recorded_at": row["recorded_at"],
            "recorded_cut": row["recorded_cut"],
            "schedule": json.loads(row["data"]),
        }
        for row in rows
    ]


def correct_schedule(
    db: Connection, trial: Trial, unit_id: str, correction: ScheduleCorrection
) -> dict:
    if trial.phase == "closed":
        require_phase(trial, "open")
    units = load_units(db, trial.trial_id)
    unit = units.get(unit_id)
    if unit is None:
        raise missing("crossover_unit", unit_id)
    if unit.revision != correction.expected_revision:
        raise conflict(
            "schedule_revision",
            "Read the current schedule before correcting it",
            details={"current_revision": unit.revision},
        )
    sites = {
        row["site_id"]: Site.model_validate_json(row["data"])
        for row in db.execute(
            "SELECT site_id, data FROM site WHERE trial_id=?", (trial.trial_id,)
        )
    }
    planned = CrossoverEnrollment(
        unit_id=unit.unit_id,
        site_id=unit.site_id,
        sequence=unit.sequence,
        load=unit.load,
        **correction.schedule_payload(),
    )
    validate_crossover_unit(trial.plan, sites, planned)
    cut = current_cut(db, trial.trial_id)
    next_revision = unit.revision + 1
    recorded_at = utc_now()
    result = db.execute(
        """INSERT INTO crossover_schedule
           (trial_id, unit_id, revision, reason, recorded_at, recorded_cut, data)
           VALUES (?,?,?,?,?,?,?)""",
        (
            trial.trial_id, unit_id, next_revision,
            correction.reason, recorded_at, cut, encoded(correction.schedule_payload()),
        ),
    )
    document = {
        "unit_id": unit.unit_id,
        "site_id": unit.site_id,
        "sequence": unit.sequence,
        "load": unit.load,
        "schedule_revision": next_revision,
    }
    db.execute(
        "UPDATE crossover_unit SET schedule_seq=?, data=? WHERE trial_id=? AND unit_id=?",
        (result.lastrowid, encoded(document), trial.trial_id, unit_id),
    )
    if trial.phase == "draft":
        db.execute(
            "UPDATE trial SET version=version+1 WHERE trial_id=?", (trial.trial_id,)
        )
    return {
        "unit_id": unit_id,
        "schedule_seq": result.lastrowid,
        "schedule_revision": next_revision,
        "recorded_cut": cut,
        "recorded_at": recorded_at,
        "reason": correction.reason,
        **correction.schedule_payload(),
    }


def _placement(db: Connection, trial: Trial, unit: CrossoverUnit) -> tuple[str, str]:
    site = Site.model_validate_json(
        db.execute(
            "SELECT data FROM site WHERE trial_id=? AND site_id=?",
            (trial.trial_id, unit.site_id),
        ).fetchone()["data"]
    )
    return site.market_id, f"{site.climate}/{unit.load}"


def schedule_report(db: Connection, trial: Trial) -> dict:
    minimum = trial.plan.crossover.policy.minimum_washout_days
    entries = []
    for unit in load_units(db, trial.trial_id).values():
        market, cell = _placement(db, trial, unit)
        entries.append(
            {
                "unit_id": unit.unit_id,
                "site_id": unit.site_id,
                "market_id": market,
                "cell": cell,
                "sequence": unit.sequence,
                "load": unit.load,
                "schedule_revision": unit.revision,
                **unit.schedule.payload(),
                "washout_days": unit.schedule.washout_days(),
                "washout_ok": unit.schedule.washout_days() >= minimum,
            }
        )
    return {"trial_id": trial.trial_id, "units": entries}


def open_checks(db: Connection, trial: Trial) -> dict:
    """开启前核对每台设备的两期+洗脱日程，以及每个单元两种顺序的配置。"""
    require_crossover(trial)
    sites = {
        row["site_id"]: Site.model_validate_json(row["data"])
        for row in db.execute(
            "SELECT site_id, data FROM site WHERE trial_id=?", (trial.trial_id,)
        )
    }
    minimum = trial.plan.crossover.policy.minimum_washout_days
    units = load_units(db, trial.trial_id)
    schedule_gaps = []
    counts: dict[tuple[str, str, str], int] = {}
    for unit in units.values():
        site = sites.get(unit.site_id)
        if site is None:
            schedule_gaps.append({"unit_id": unit.unit_id, "reason": "missing_site"})
            continue
        market = trial.plan.market(site.market_id)
        cell = f"{site.climate}/{unit.load}"
        if market is None or not any(item.key == cell for item in market.cells):
            schedule_gaps.append({"unit_id": unit.unit_id, "reason": "unplanned_cell"})
            continue
        counts[(site.market_id, cell, unit.sequence)] = (
            counts.get((site.market_id, cell, unit.sequence), 0) + 1
        )
        if unit.schedule.washout_days() < minimum:
            schedule_gaps.append(
                {
                    "unit_id": unit.unit_id,
                    "reason": "washout_too_short",
                    "observed": unit.schedule.washout_days(),
                    "required": minimum,
                }
            )
        if (
            unit.schedule.first_from < trial.plan.starts_on
            or unit.schedule.second_until > trial.plan.ends_before
        ):
            schedule_gaps.append(
                {"unit_id": unit.unit_id, "reason": "schedule_outside_trial"}
            )
    required = trial.plan.crossover.policy.required_units_per_sequence
    allocation_gaps = []
    for market in trial.plan.markets:
        for planned_cell in market.cells:
            for sequence in SEQUENCES:
                observed = counts.get((market.market_id, planned_cell.key, sequence), 0)
                if observed < required:
                    allocation_gaps.append(
                        {
                            "market": market.market_id,
                            "cell": planned_cell.key,
                            "sequence": sequence,
                            "enrolled": observed,
                            "required": required,
                        }
                    )
    return {
        "ready": not schedule_gaps and not allocation_gaps,
        "schedule_gaps": schedule_gaps,
        "allocation_gaps": allocation_gaps,
        "enrolled_units": len(units),
    }


def open_crossover(db: Connection, trial: Trial, expected: int) -> Trial:
    check_version(trial, expected)
    require_phase(trial, "draft")
    checks = open_checks(db, trial)
    if not checks["ready"]:
        raise conflict(
            "crossover_allocation_shortfall",
            "Crossover schedules or sequence allocation are not ready",
            details=checks,
        )
    db.execute(
        "UPDATE trial SET phase='open', version=version+1 WHERE trial_id=?",
        (trial.trial_id,),
    )
    from fieldproof.design.registry import get_trial

    return get_trial(db, trial.trial_id)
