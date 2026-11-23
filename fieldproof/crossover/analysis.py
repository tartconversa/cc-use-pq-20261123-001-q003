"""两期交叉试验的成效预览、配对统计与放行判断。"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date
from fractions import Fraction
from sqlite3 import Connection

from fieldproof.analysis.contrast import display
from fieldproof.analysis.cohorts import AnalysisWindow
from fieldproof.common import fingerprint, missing
from fieldproof.crossover.periods import (
    ClassifiedCycle,
    classify,
)
from fieldproof.crossover.registry import (
    SEQUENCES,
    CrossoverUnit,
    UnitSchedule,
)
from fieldproof.design.contracts import Site
from fieldproof.design.registry import Roster, Trial
from fieldproof.signals.cycles import Cycle, resolve_cycles
from fieldproof.signals.ledger import current_cut, events_at, require_cut

RATE_METRICS = ("manual", "alarm", "failure")
METRICS = (*RATE_METRICS, "energy_wh", "duration_seconds")
PERIODS = (1, 2)
EVIDENCE_FORMAT = 2


@dataclass
class PeriodStats:
    """一台设备在一个分配期内的观察（只含被该期接受的完整/中止周期）。"""

    usable_cycles: int = 0
    completed: int = 0
    failed: int = 0
    manual: int = 0
    alarms: int = 0
    energy_cycles: int = 0
    total_wh: int = 0
    elapsed_seconds: int = 0
    energy_issues: Counter = field(default_factory=Counter)

    def record(self, cycle: Cycle):
        self.usable_cycles += 1
        self.completed += int(not cycle.failed)
        self.failed += int(cycle.failed)
        self.manual += int(cycle.manual)
        self.alarms += int(cycle.alarm)
        self.elapsed_seconds += cycle.elapsed_seconds
        if cycle.energy.wh is not None:
            self.energy_cycles += 1
            self.total_wh += cycle.energy.wh
        elif not cycle.failed:
            self.energy_issues[cycle.energy.problem] += 1

    def rate(self, numerator: int) -> Fraction | None:
        return Fraction(numerator, self.usable_cycles) if self.usable_cycles else None

    def metric(self, name: str) -> Fraction | None:
        if name == "energy_wh":
            return Fraction(self.total_wh, self.energy_cycles) if self.energy_cycles else None
        if name == "duration_seconds":
            return (
                Fraction(self.elapsed_seconds, self.usable_cycles)
                if self.usable_cycles
                else None
            )
        numerators = {"manual": self.manual, "alarm": self.alarms, "failure": self.failed}
        return self.rate(numerators[name])

    def energy_coverage(self) -> Fraction | None:
        return Fraction(self.energy_cycles, self.completed) if self.completed else None

    def eligible(self, required_cycles: int, max_missing_energy_bp: int) -> bool:
        if self.usable_cycles < required_cycles or self.completed == 0:
            return False
        coverage = self.energy_coverage()
        if coverage is None:
            return False
        return coverage * 10_000 >= 10_000 - max_missing_energy_bp

    def payload(self) -> dict:
        return {
            "usable_cycles": self.usable_cycles,
            "completed_cycles": self.completed,
            "failed_cycles": self.failed,
            "manual_cycles": self.manual,
            "alarm_cycles": self.alarms,
            "energy_cycles": self.energy_cycles,
            "total_wh": self.total_wh,
            "elapsed_seconds": self.elapsed_seconds,
            "energy_issues": dict(sorted(self.energy_issues.items())),
            "energy_coverage_bp": display(self.energy_coverage(), 10_000),
        }


@dataclass
class UnitRow:
    unit: CrossoverUnit
    market: str
    cell: str
    periods: dict[int, PeriodStats] = field(default_factory=lambda: {1: PeriodStats(), 2: PeriodStats()})
    integrity_issues: Counter = field(default_factory=Counter)
    unresolved_alarm_cycles: int = 0
    outside_cycles: list[ClassifiedCycle] = field(default_factory=list)
    all_cycles: list[ClassifiedCycle] = field(default_factory=list)

    def period_eligible(self, trial: Trial, period: int) -> bool:
        policy = trial.plan.crossover.policy
        return self.periods[period].eligible(
            policy.required_cycles_per_period, policy.max_missing_energy_bp
        )

    def pairing(self, trial: Trial) -> str:
        first = self.period_eligible(trial, 1)
        second = self.period_eligible(trial, 2)
        if first and second:
            return "paired"
        if first:
            return "period1_only"
        if second:
            return "period2_only"
        return "neither_period"

    def contrast(self, name: str) -> Fraction | None:
        """同设备辅助期减标准期的设备内效果。"""
        assisted_period = 2 if self.unit.sequence == "ab" else 1
        standard_period = 1 if assisted_period == 2 else 2
        assisted = self.periods[assisted_period].metric(name)
        standard = self.periods[standard_period].metric(name)
        if assisted is None or standard is None:
            return None
        return assisted - standard

    def period_arm(self, period: int) -> str:
        return self.unit.arm_in(period)


@dataclass(frozen=True)
class CrossoverMaterial:
    window: AnalysisWindow
    cut: int
    schedule_seq: int
    rows: dict[str, UnitRow]
    classified: list[ClassifiedCycle]
    sites: dict[str, Site]
    units: dict[str, CrossoverUnit]


# ---------- 载入 ----------


def current_schedule_seq(db: Connection, trial_id: str) -> int:
    return db.execute(
        "SELECT COALESCE(MAX(seq),0) FROM crossover_schedule WHERE trial_id=?",
        (trial_id,),
    ).fetchone()[0]


def _units_as_of(db: Connection, trial: Trial, schedule_seq: int) -> dict[str, CrossoverUnit]:
    rows = db.execute(
        "SELECT data, unit_id FROM crossover_unit WHERE trial_id=? ORDER BY unit_id",
        (trial.trial_id,),
    ).fetchall()
    units = {}
    for row in rows:
        base = json.loads(row["data"])
        chosen = db.execute(
            """SELECT data, revision FROM crossover_schedule
               WHERE trial_id=? AND unit_id=? AND seq<=?
               ORDER BY revision DESC LIMIT 1""",
            (trial.trial_id, row["unit_id"], schedule_seq),
        ).fetchone()
        if chosen is None:
            # The schedule ledger position predates this unit's first revision.
            # The roster is fixed at opening, so keep the unit and replay its
            # earliest known schedule rather than silently dropping it.
            chosen = db.execute(
                """SELECT data, revision FROM crossover_schedule
                   WHERE trial_id=? AND unit_id=? ORDER BY seq ASC LIMIT 1""",
                (trial.trial_id, row["unit_id"]),
            ).fetchone()
        if chosen is None:
            continue
        schedule = UnitSchedule.from_payload(json.loads(chosen["data"]))
        units[row["unit_id"]] = CrossoverUnit(
            base["unit_id"], base["site_id"], base["sequence"], base["load"],
            chosen["revision"], schedule,
        )
    return units


def _roster_as_of(units: dict[str, CrossoverUnit], sites: dict[str, Site]) -> Roster:
    from fieldproof.design.contracts import Enrollment

    enrollments = {
        unit.unit_id: Enrollment(
            unit_id=unit.unit_id,
            site_id=unit.site_id,
            arm=unit.arm_in(1),
            load=unit.load,
            active_from=unit.schedule.first_from,
            active_until=unit.schedule.second_until,
        )
        for unit in units.values()
    }
    return Roster(sites, enrollments)


def load_sites(db: Connection, trial: Trial) -> dict[str, Site]:
    return {
        row["site_id"]: Site.model_validate_json(row["data"])
        for row in db.execute(
            "SELECT site_id, data FROM site WHERE trial_id=?", (trial.trial_id,)
        )
    }


def load(
    db: Connection,
    trial: Trial,
    window: AnalysisWindow,
    cut: int | None,
    schedule_seq: int | None = None,
) -> CrossoverMaterial:
    selected_cut = current_cut(db, trial.trial_id) if cut is None else cut
    require_cut(db, trial.trial_id, selected_cut)
    selected_schedule = (
        current_schedule_seq(db, trial.trial_id) if schedule_seq is None else schedule_seq
    )
    if selected_schedule and not db.execute(
        "SELECT 1 FROM crossover_schedule WHERE trial_id=? AND seq=? LIMIT 1",
        (trial.trial_id, selected_schedule),
    ).fetchone():
        raise missing("schedule_seq", str(selected_schedule))
    sites = load_sites(db, trial)
    units = _units_as_of(db, trial, selected_schedule)
    registered = db.execute(
        "SELECT COUNT(*) FROM crossover_unit WHERE trial_id=?", (trial.trial_id,)
    ).fetchone()[0]
    if registered and not units:
        raise missing("schedule_seq", str(selected_schedule))
    roster = _roster_as_of(units, sites)
    cycles = resolve_cycles(
        events_at(db, trial.trial_id, selected_cut), roster, trial.plan.policy
    )
    classified = classify(cycles, units, sites)
    classified = [
        item
        for item in classified
        if item.start_day is None or window.contains(item.start_day)
    ]
    rows: dict[str, UnitRow] = {}
    for unit in units.values():
        site = sites[unit.site_id]
        market = trial.plan.market(site.market_id)
        rows[unit.unit_id] = UnitRow(
            unit, site.market_id, f"{site.climate}/{unit.load}"
        )
    for item in classified:
        row = rows.get(item.unit_id)
        if row is None:
            continue
        row.all_cycles.append(item)
        if not item.cycle.usable:
            row.integrity_issues.update(item.cycle.issues)
            row.unresolved_alarm_cycles += int(item.cycle.alarm)
            continue
        if item.accepted:
            row.periods[item.accepted_period].record(item.cycle)
        else:
            row.outside_cycles.append(item)
    return CrossoverMaterial(
        window, selected_cut, selected_schedule, rows, classified, sites, units
    )


# ---------- 聚合 ----------


@dataclass
class SequenceAggregate:
    sequence: str
    recruited: list[UnitRow]
    paired: list[UnitRow]
    period1_only: int
    period2_only: int
    neither: int

    @classmethod
    def build(cls, sequence: str, rows: list[UnitRow], trial: Trial):
        recruited = rows
        paired = [row for row in rows if row.pairing(trial) == "paired"]
        statuses = Counter(row.pairing(trial) for row in rows)
        return cls(
            sequence,
            recruited,
            paired,
            statuses["period1_only"],
            statuses["period2_only"],
            statuses["neither_period"],
        )

    def effect(self, name: str) -> Fraction | None:
        values = [value for row in self.paired if (value := row.contrast(name)) is not None]
        return sum(values, Fraction(0)) / len(values) if values else None

    def period_arm_mean(self, period: int, name: str) -> Fraction | None:
        values = [
            value
            for row in self.paired
            if (value := row.periods[period].metric(name)) is not None
        ]
        return sum(values, Fraction(0)) / len(values) if values else None

    def coverage(self) -> Fraction:
        return Fraction(len(self.paired), len(self.recruited)) if self.recruited else Fraction(0)

    def payload(self, trial: Trial) -> dict:
        return {
            "sequence": self.sequence,
            "recruited_units": len(self.recruited),
            "paired_units": len(self.paired),
            "period1_only_units": self.period1_only,
            "period2_only_units": self.period2_only,
            "neither_period_units": self.neither,
            "pairing_coverage_bp": display(self.coverage(), 10_000),
            "periods": {
                str(period): {
                    "arm": _arm(self.sequence, period),
                    "sample_units": len(self.paired),
                    "metrics": {
                        name: display(self.period_arm_mean(period, name), _scale(name))
                        for name in METRICS
                    },
                }
                for period in PERIODS
            },
            "effects": {
                name: display(self.effect(name), _scale(name)) for name in METRICS
            },
        }


def _arm(sequence: str, period: int) -> str:
    if sequence == "ab":
        return "standard" if period == 1 else "assisted"
    return "assisted" if period == 1 else "standard"


def _scale(name: str) -> int:
    return 10_000 if name in RATE_METRICS else 1


@dataclass
class CellAggregate:
    market_id: str
    key: str
    weight_bp: int
    sequences: dict[str, SequenceAggregate]
    gaps: list[dict]

    def paired_rows(self) -> list[UnitRow]:
        paired = []
        for aggregate in self.sequences.values():
            paired.extend(aggregate.paired)
        return paired

    def merged_effect(self, name: str) -> Fraction | None:
        values = [value for row in self.paired_rows() if (value := row.contrast(name)) is not None]
        return sum(values, Fraction(0)) / len(values) if values else None

    def supported(self, trial: Trial) -> bool:
        required = trial.plan.crossover.policy.required_units_per_sequence
        return all(
            len(self.sequences.get(sequence, SequenceAggregate(sequence, [], [], 0, 0, 0)).paired)
            >= required
            for sequence in SEQUENCES
        ) and not self.gaps


def _cell_gaps(cell: CellAggregate, trial: Trial) -> list[dict]:
    required = trial.plan.crossover.policy.required_units_per_sequence
    gaps = []
    for sequence in SEQUENCES:
        aggregate = cell.sequences.get(sequence)
        paired = len(aggregate.paired) if aggregate else 0
        recruited = len(aggregate.recruited) if aggregate else 0
        if recruited == 0:
            gaps.append({"sequence": sequence, "reason": "sequence_missing", "observed": 0,
                         "required": required})
        elif paired < required:
            gaps.append({"sequence": sequence, "reason": "paired_units", "observed": paired,
                         "required": required})
    return gaps


def build_cells(trial: Trial, rows: dict[str, UnitRow]) -> list[CellAggregate]:
    grouped: dict[tuple[str, str], list[UnitRow]] = defaultdict(list)
    for row in rows.values():
        grouped[(row.market, row.cell)].append(row)
    cells = []
    for market in trial.plan.markets:
        for planned in market.cells:
            cell_rows = grouped.get((market.market_id, planned.key), [])
            by_sequence = {
                sequence: SequenceAggregate.build(
                    sequence,
                    [row for row in cell_rows if row.unit.sequence == sequence],
                    trial,
                )
                for sequence in SEQUENCES
            }
            aggregate = CellAggregate(
                market.market_id, planned.key, planned.weight_bp, by_sequence, []
            )
            aggregate.gaps.extend(_cell_gaps(aggregate, trial))
            cells.append(aggregate)
    return cells


# ---------- 护栏 ----------


@dataclass(frozen=True)
class Gate:
    name: str
    state: str
    observed: str | None
    boundary: int | None
    direction: str | None
    details: dict | None = None

    def payload(self) -> dict:
        data = {
            "gate": self.name,
            "state": self.state,
            "observed": self.observed,
            "boundary": self.boundary,
            "direction": self.direction,
        }
        if self.details:
            data["details"] = self.details
        return data


def _weighted_effect(trial: Trial, cells: list[CellAggregate], name: str) -> Fraction | None:
    total = Fraction(0)
    for cell in cells:
        if not cell.supported(trial):
            return None
        value = cell.merged_effect(name)
        if value is None:
            return None
        total += value * Fraction(cell.weight_bp, 10_000)
    return total


def _divergence(cells: list[CellAggregate], name: str) -> Fraction | None:
    worst = None
    for cell in cells:
        ab = cell.sequences.get("ab")
        ba = cell.sequences.get("ba")
        if ab is None or ba is None:
            return None
        left, right = ab.effect(name), ba.effect(name)
        if left is None or right is None:
            return None
        spread = abs(left - right)
        worst = spread if worst is None else max(worst, spread)
    return worst


def assess_market(trial: Trial, market_id: str, all_cells: list[CellAggregate],
                  material: CrossoverMaterial) -> dict:
    market = trial.plan.market(market_id)
    if market is None:
        raise missing("market", market_id)
    cells = [
        cell for cell in all_cells
        if cell.market_id == market_id and cell.key in {item.key for item in market.cells}
    ]
    policy = trial.plan.crossover.policy
    base_policy = trial.plan.policy
    gates: list[Gate] = []

    coverage_gaps = [gap for cell in cells for gap in cell.gaps]
    gates.append(Gate(
        "crossover_sequence_coverage",
        "pass" if not coverage_gaps else "unavailable",
        None, None, None,
        details={"gaps": coverage_gaps} if coverage_gaps else None,
    ))

    market_rows = [row for row in material.rows.values() if row.market == market_id]
    unresolved_units = sum(bool(row.integrity_issues) for row in market_rows)
    gates.append(Gate(
        "units_with_unresolved_cycles",
        "pass" if unresolved_units == 0 else "fail",
        str(unresolved_units), 0, "at_most",
    ))

    transition_alarms = sum(
        1
        for row in market_rows
        for item in row.outside_cycles
        if item.transition_alarm
    )
    gates.append(Gate(
        "transition_alarms_visible",
        "pass" if transition_alarms == 0 else "fail",
        str(transition_alarms), 0, "at_most",
    ))

    manual = _weighted_effect(trial, cells, "manual")
    gates.append(Gate(
        "manual_saving_bp",
        "unavailable" if manual is None else (
            "pass" if -manual * 10_000 >= base_policy.min_manual_saving_bp else "fail"
        ),
        None if manual is None else display(-manual, 10_000),
        base_policy.min_manual_saving_bp, "at_least",
    ))
    for name, boundary in (
        ("alarm", base_policy.max_alarm_increase_bp),
        ("failure", base_policy.max_failure_increase_bp),
    ):
        effect = _weighted_effect(trial, cells, name)
        gates.append(Gate(
            f"{name}_increase_bp",
            "unavailable" if effect is None else (
                "pass" if effect * 10_000 <= boundary else "fail"
            ),
            None if effect is None else display(effect, 10_000),
            boundary, "at_most",
        ))

    energy_effect = _weighted_effect(trial, cells, "energy_wh")
    standard_energy = _weighted_period(trial, cells, "standard", "energy_wh")
    if energy_effect is None or standard_energy is None:
        gates.append(Gate("energy_change_bp", "unavailable", None,
                          base_policy.max_energy_increase_bp, "at_most"))
    elif standard_energy == 0:
        gates.append(Gate(
            "zero_baseline_energy",
            "pass" if energy_effect == 0 else "fail",
            display(energy_effect), 0, "at_most",
        ))
    else:
        relative = energy_effect / standard_energy * 10_000
        gates.append(Gate(
            "energy_change_bp",
            "pass" if relative <= base_policy.max_energy_increase_bp else "fail",
            display(relative), base_policy.max_energy_increase_bp, "at_most",
        ))

    for name in RATE_METRICS:
        spread = _divergence(cells, name)
        gates.append(Gate(
            f"sequence_divergence_{name}_bp",
            "unavailable" if spread is None else (
                "pass" if spread * 10_000 <= policy.max_sequence_divergence_bp else "fail"
            ),
            None if spread is None else display(spread, 10_000),
            policy.max_sequence_divergence_bp, "at_most",
            details=_divergence_details(cells, name, 10_000),
        ))
    energy_spread = _divergence(cells, "energy_wh")
    gates.append(Gate(
        "sequence_divergence_energy_wh",
        "unavailable" if energy_spread is None else (
            "pass" if energy_spread <= policy.max_energy_divergence_wh else "fail"
        ),
        None if energy_spread is None else display(energy_spread),
        policy.max_energy_divergence_wh, "at_most",
        details=_divergence_details(cells, "energy_wh", 1),
    ))

    if any(gate.state == "fail" for gate in gates):
        recommendation = "hold"
    elif any(gate.state != "pass" for gate in gates):
        recommendation = "insufficient"
    else:
        recommendation = "advance"
    return {
        "recommendation": recommendation,
        "gates": [gate.payload() for gate in gates],
        "blocked_by": [gate.name for gate in gates if gate.state != "pass"],
    }


def _weighted_period(trial: Trial, cells: list[CellAggregate], arm: str, name: str) -> Fraction | None:
    total = Fraction(0)
    for cell in cells:
        if not cell.supported(trial):
            return None
        contributions = []
        for sequence in SEQUENCES:
            aggregate = cell.sequences.get(sequence)
            if aggregate is None:
                return None
            for period in PERIODS:
                if _arm(sequence, period) == arm:
                    value = aggregate.period_arm_mean(period, name)
                    if value is None:
                        return None
                    contributions.append(value)
        total += (sum(contributions, Fraction(0)) / len(contributions)) * Fraction(
            cell.weight_bp, 10_000
        )
    return total


def _divergence_details(cells: list[CellAggregate], name: str, scale: int) -> dict:
    per_cell = []
    for cell in cells:
        ab = cell.sequences.get("ab")
        ba = cell.sequences.get("ba")
        ab_effect = ab.effect(name) if ab else None
        ba_effect = ba.effect(name) if ba else None
        spread = (
            abs(ab_effect - ba_effect)
            if ab_effect is not None and ba_effect is not None
            else None
        )
        per_cell.append({
            "cell": cell.key,
            "ab_effect": display(ab_effect, scale),
            "ba_effect": display(ba_effect, scale),
            "spread": display(spread, scale),
        })
    return {
        "metric": name,
        "by_cell": per_cell,
        "note": "worst absolute spread across planned cells",
    }


# ---------- 文档 ----------


def _unit_payload(row: UnitRow, trial: Trial) -> dict:
    return {
        "unit_id": row.unit.unit_id,
        "site_id": row.unit.site_id,
        "sequence": row.unit.sequence,
        "cell": row.cell,
        "schedule_revision": row.unit.revision,
        "pairing": row.pairing(trial),
        "periods": {
            str(period): {
                "arm": row.period_arm(period),
                "eligible": row.period_eligible(trial, period),
                **row.periods[period].payload(),
            }
            for period in PERIODS
        },
        "contrasts": {
            name: display(row.contrast(name), _scale(name)) for name in METRICS
        },
        "integrity_issues": dict(sorted(row.integrity_issues.items())),
        "unresolved_alarm_cycles": row.unresolved_alarm_cycles,
        "outside_cycles": [item.payload() for item in row.outside_cycles],
    }


def analyze(trial: Trial, material: CrossoverMaterial) -> dict:
    cells = build_cells(trial, material.rows)
    market_results = []
    for market in trial.plan.markets:
        keys = {cell.key for cell in market.cells}
        market_cells = [
            cell for cell in cells
            if cell.market_id == market.market_id and cell.key in keys
        ]
        assessment = assess_market(trial, market.market_id, cells, material)
        market_results.append({
            "market_id": market.market_id,
            "label": market.label,
            "cells": [
                {
                    "cell": cell.key,
                    "weight_bp": cell.weight_bp,
                    "gaps": cell.gaps,
                    "sequences": [
                        cell.sequences[sequence].payload(trial) for sequence in SEQUENCES
                    ],
                    "merged_effects": {
                        name: display(cell.merged_effect(name), _scale(name)) for name in METRICS
                    },
                }
                for cell in market_cells
            ],
            **assessment,
        })
    units = [_unit_payload(row, trial) for row in sorted(material.rows.values(),
                                                        key=lambda item: item.unit.unit_id)]
    relevant = [item.payload() for item in material.classified]
    return {
        "format": EVIDENCE_FORMAT,
        "design": "crossover",
        "trial_id": trial.trial_id,
        "program": trial.plan.program,
        "cut": material.cut,
        "schedule_seq": material.schedule_seq,
        "window": material.window.payload(),
        "policy": trial.plan.policy.model_dump(),
        "crossover_policy": trial.plan.crossover.policy.model_dump(),
        "roster_digest": fingerprint([
            (row.unit.unit_id, row.unit.site_id, row.market, row.cell,
             row.unit.sequence, row.unit.revision)
            for row in material.rows.values()
        ]),
        "facts_digest": fingerprint(relevant),
        "unit_digest": fingerprint(units),
        "market_results": market_results,
        "units": units,
    }


def build(db: Connection, trial: Trial, window: AnalysisWindow, cut: int | None,
          schedule_seq: int | None = None) -> dict:
    material = load(db, trial, window, cut, schedule_seq)
    document = analyze(trial, material)
    return {"evidence_id": fingerprint(document), **document}


# ---------- 现场核查队列 ----------


def inspection_queue(
    trial: Trial, material: CrossoverMaterial, market: str, site: str | None,
    offset: int, limit: int,
) -> dict:
    if trial.plan.market(market) is None:
        raise missing("market", market)
    priorities = {
        "unresolved_alarm": 0,
        "transition_alarm": 1,
        "integrity": 2,
        "pending": 3,
        "outside_period": 4,
    }
    entries = []
    for item in material.classified:
        row = material.rows.get(item.unit_id)
        if row is None or row.market != market:
            continue
        if site is not None and row.unit.site_id != site:
            continue
        category = None
        issues = list(item.cycle.issues)
        if not item.cycle.usable:
            category = "unresolved_alarm" if item.cycle.alarm else (
                "pending" if item.cycle.state == "pending" else "integrity"
            )
        elif not item.accepted:
            category = "transition_alarm" if item.transition_alarm else "outside_period"
            issues = [item.outside_reason]
        if category is None:
            continue
        entries.append({
            "category": category,
            "unit_id": item.unit_id,
            "cycle_id": item.cycle_id,
            "local_date": item.start_day.isoformat() if item.start_day else None,
            "site": row.unit.site_id,
            "sequence": row.unit.sequence,
            "cell": row.cell,
            "nominal_period": item.nominal_period,
            "accepted_period": item.accepted_period,
            "issues": issues,
            "source_revisions": item.cycle.sources,
        })
    entries.sort(key=lambda item: (
        priorities[item["category"]], item["local_date"] or "",
        item["unit_id"], item["cycle_id"],
    ))
    totals = Counter(item["category"] for item in entries)
    return {
        "market_id": market,
        "cut": material.cut,
        "schedule_seq": material.schedule_seq,
        "window": material.window.payload(),
        "total": len(entries),
        "counts": {category: totals[category] for category in priorities},
        "items": entries[offset:offset + limit],
        "next_offset": offset + limit if offset + limit < len(entries) else None,
    }


# ---------- 覆盖与流失 ----------


def coverage_report(trial: Trial, material: CrossoverMaterial, market: str) -> dict:
    if trial.plan.market(market) is None:
        raise missing("market", market)
    cells = build_cells(trial, material.rows)
    payload_cells = []
    for cell in cells:
        if cell.market_id != market:
            continue
        payload_cells.append({
            "cell": cell.key,
            "sequences": [
                {
                    "sequence": sequence,
                    "recruited_units": len(aggregate.recruited),
                    "paired_units": len(aggregate.paired),
                    "period1_only_units": aggregate.period1_only,
                    "period2_only_units": aggregate.period2_only,
                    "neither_period_units": aggregate.neither,
                    "pairing_coverage_bp": display(aggregate.coverage(), 10_000),
                }
                for sequence, aggregate in cell.sequences.items()
            ],
        })
    return {
        "market_id": market,
        "cut": material.cut,
        "schedule_seq": material.schedule_seq,
        "cells": payload_cells,
    }
