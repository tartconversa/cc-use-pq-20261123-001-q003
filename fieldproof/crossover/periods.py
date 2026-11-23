"""把周期归入两期本地日程，并识别跨期/洗脱的可疑数据。"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from zoneinfo import ZoneInfo

from fieldproof.crossover.registry import CrossoverUnit, UnitSchedule
from fieldproof.design.contracts import Site, local_day
from fieldproof.signals.cycles import Cycle

# 周期无法成为某一期的有效配对观察时的原因代码。
BEFORE_TRIAL = "before_first_period"
AFTER_TRIAL = "after_second_period"
WASHOUT = "washout_period"
CROSSES_BOUNDARY = "crosses_period_boundary"
EXCLUSION_REASONS = (BEFORE_TRIAL, AFTER_TRIAL, WASHOUT, CROSSES_BOUNDARY)


@dataclass(frozen=True)
class ClassifiedCycle:
    unit_id: str
    cycle_id: str
    start_day: date | None
    close_day: date | None
    nominal_period: int | None
    accepted_period: int | None
    outside_reason: str | None
    transition_alarm: bool
    cycle: Cycle

    @property
    def accepted(self) -> bool:
        return self.accepted_period is not None

    def payload(self) -> dict:
        return {
            "unit_id": self.unit_id,
            "cycle_id": self.cycle_id,
            "local_date": self.start_day.isoformat() if self.start_day else None,
            "close_local_date": self.close_day.isoformat() if self.close_day else None,
            "state": self.cycle.state,
            "nominal_period": self.nominal_period,
            "accepted_period": self.accepted_period,
            "outside_reason": self.outside_reason,
            "transition_alarm": self.transition_alarm,
            "alarm": self.cycle.alarm,
            "manual": self.cycle.manual,
            "issues": self.cycle.issues,
            "source_revisions": self.cycle.sources,
        }


def _local_days(cycle: Cycle, unit: CrossoverUnit, sites: dict[str, Site]) -> tuple[date | None, date | None]:
    site = sites[unit.site_id]
    zone = ZoneInfo(site.timezone)
    start_day = local_day(cycle.start.occurred_at, site) if cycle.start else None
    close_day = local_day(cycle.close.occurred_at, site) if cycle.close else None
    return start_day, close_day


def _reason(schedule: UnitSchedule, start_day: date, close_day: date) -> tuple[int | None, str | None]:
    """完整闭合区间必须落在同一分配期内，不能只凭开始日期。"""
    for period in (1, 2):
        lower, upper = schedule.bounds(period)
        if lower <= start_day and close_day < upper:
            return period, None
    if close_day < schedule.first_from:
        return None, BEFORE_TRIAL
    if start_day >= schedule.second_until:
        return None, AFTER_TRIAL
    if start_day >= schedule.first_until and close_day <= schedule.second_from:
        return None, WASHOUT
    return None, CROSSES_BOUNDARY


def classify_cycle(
    cycle: Cycle, unit: CrossoverUnit, sites: dict[str, Site]
) -> ClassifiedCycle:
    start_day, close_day = _local_days(cycle, unit, sites)
    nominal = unit.schedule.period_starting_on(start_day) if start_day else None
    if not cycle.usable or start_day is None or close_day is None:
        return ClassifiedCycle(
            cycle.unit_id, cycle.cycle_id, start_day, close_day, nominal,
            None, None, False, cycle,
        )
    accepted_period, reason = _reason(unit.schedule, start_day, close_day)
    transition = cycle.alarm and reason is not None
    return ClassifiedCycle(
        cycle.unit_id, cycle.cycle_id, start_day, close_day, nominal,
        accepted_period, reason, transition, cycle,
    )


def classify(
    cycles: list[Cycle],
    units: dict[str, CrossoverUnit],
    sites: dict[str, Site],
) -> list[ClassifiedCycle]:
    results = []
    for cycle in cycles:
        unit = units.get(cycle.unit_id)
        if unit is not None:
            results.append(classify_cycle(cycle, unit, sites))
    return results
