from tests.support import (
    CROSS_ROOT,
    CrossoverCase,
    cross_observations,
    crossover_plan,
    crossover_unit,
)


def two_cell_plan():
    plan_data = crossover_plan()
    plan_data["markets"] = [{
        "market_id": "alpine", "label": "Alpine",
        "cells": [
            {"climate": "cold", "load": "light", "weight_bp": 7_000},
            {"climate": "cold", "load": "heavy", "weight_bp": 3_000},
        ],
    }]
    return plan_data


class CrossoverEdgeTest(CrossoverCase):
    def test_market_weights_combine_cells_with_sequence_effects_reported_per_cell(self):
        self.create_crossover(two_cell_plan())
        self.register_site()
        units = []
        for sequence in ("ab", "ba"):
            for load in ("light", "heavy"):
                for number in (1, 2):
                    units.append(crossover_unit(
                        f"{sequence}-{load}-{number}", sequence, load=load
                    ))
        result = self.enroll_crossover(units)
        self.open_crossover(result["version"])
        rows = []
        for unit in units:
            is_ab = unit["sequence"] == "ab"
            standard_manual = is_ab  # AB sees manual in period 1; BA in period 2
            rows += cross_observations(
                unit["unit_id"], "p1", "2026-10-25T09:00:00+00:00",
                manual=standard_manual, wh=100,
            )
            rows += cross_observations(
                unit["unit_id"], "p2", "2026-11-01T09:00:00+00:00",
                manual=not standard_manual, wh=90,
            )
        self.cross_deliver("batch", rows)
        market = self.cross_market()
        self.assertEqual([cell["cell"] for cell in market["cells"]],
                         ["cold/light", "cold/heavy"])
        for cell in market["cells"]:
            for block in cell["sequences"]:
                self.assertEqual(block["paired_units"], 2)
        # Every unit shows a full manual saving, so weights do not change direction.
        light = next(cell for cell in market["cells"] if cell["cell"] == "cold/light")
        heavy = next(cell for cell in market["cells"] if cell["cell"] == "cold/heavy")
        self.assertEqual(light["merged_effects"]["manual"], "-10000.000")
        self.assertEqual(heavy["merged_effects"]["manual"], "-10000.000")
        self.assertEqual(market["recommendation"], "advance")

    def test_preview_replays_schedule_as_of_a_ledger_position(self):
        self.setup_crossover()
        rows = self.balanced_events()
        # Replace ab-one's period-1 cycle with one sitting on the last period day.
        rows = [r for r in rows if not (r["unit_id"] == "ab-one" and r["cycle_id"] == "p1")]
        rows += cross_observations(
            "ab-one", "p1", "2026-10-28T12:00:00+00:00", manual=True, wh=100
        )
        cut = self.cross_deliver("batch", rows)["current_cut"]
        first_seq = self.cross_preview()["schedule_seq"]
        before = self.cross_preview(cut=cut, schedule_seq=first_seq)
        before_unit = next(item for item in before["units"] if item["unit_id"] == "ab-one")
        self.assertEqual(before_unit["pairing"], "paired")

        correction = self.call(
            "PUT", CROSS_ROOT + "/crossover/units/ab-one/schedule",
            {
                "expected_revision": 1,
                "first_from": "2026-10-24", "first_until": "2026-10-28",
                "second_from": "2026-10-30", "second_until": "2026-11-05",
                "reason": "diary",
            },
        )
        # The historical ledger position still reproduces the original judgement.
        replay = self.cross_preview(cut=cut, schedule_seq=first_seq)
        replay_unit = next(item for item in replay["units"] if item["unit_id"] == "ab-one")
        self.assertEqual(replay_unit["pairing"], "paired")
        current = self.cross_preview(cut=cut, schedule_seq=correction["schedule_seq"])
        current_unit = next(item for item in current["units"] if item["unit_id"] == "ab-one")
        self.assertEqual(current_unit["pairing"], "period2_only")

    def test_unknown_schedule_position_is_rejected(self):
        self.setup_crossover()
        self.cross_deliver("batch", self.balanced_events())
        problem = self.call(
            "GET", CROSS_ROOT + "/crossover/preview",
            params={"schedule_seq": 999_999}, status=404,
        )
        self.assertEqual(problem["code"], "schedule_seq_missing")

    def test_open_state_schedule_correction_can_make_the_market_undecidable(self):
        opened = self.setup_crossover()
        # Each unit contributes exactly one cycle per period, so losing one period
        # on one AB unit drops the AB sequence below the two-paired minimum.
        rows = self.balanced_events()
        # Move ab-one's only period-1 cycle onto the last originally-registered day.
        rows = [r for r in rows if not (r["unit_id"] == "ab-one" and r["cycle_id"] == "p1")]
        rows += cross_observations(
            "ab-one", "p1", "2026-10-28T12:00:00+00:00", manual=True, wh=100
        )
        cut = self.cross_deliver("batch", rows)["current_cut"]
        self.assertEqual(self.cross_market()["recommendation"], "advance")
        first_seq = self.cross_preview()["schedule_seq"]
        correction = self.call(
            "PUT", CROSS_ROOT + "/crossover/units/ab-one/schedule",
            {
                "expected_revision": 1,
                "first_from": "2026-10-24", "first_until": "2026-10-28",
                "second_from": "2026-10-30", "second_until": "2026-11-05",
                "reason": "diary",
            },
        )
        after = self.cross_preview(cut=cut, schedule_seq=correction["schedule_seq"])
        self.assertEqual(after["market_results"][0]["recommendation"], "insufficient")
        # The trial itself stays open and its version is untouched by the correction.
        self.assertEqual(self.call("GET", CROSS_ROOT)["version"], opened["version"])
        # The historical position still reproduces the decidable result.
        replay = self.cross_preview(cut=cut, schedule_seq=first_seq)
        self.assertEqual(replay["market_results"][0]["recommendation"], "advance")

    def test_aborted_cycle_counts_as_failure_inside_its_assigned_period(self):
        self.setup_crossover()
        rows = self.balanced_events()
        rows += cross_observations(
            "ab-two", "broken", "2026-10-26T09:00:00+00:00",
            wh=None, abort=True,
        )
        self.cross_deliver("batch", rows)
        preview = self.cross_preview()
        unit = next(item for item in preview["units"] if item["unit_id"] == "ab-two")
        first = unit["periods"]["1"]
        self.assertEqual(first["failed_cycles"], 1)
        # Abort is a usable cycle so the period still meets its one-cycle minimum.
        self.assertTrue(first["eligible"])
        # The abort sits in ab-two's standard period; its assisted period has no
        # failure, so the within-unit failure effect is negative (assisted better).
        self.assertEqual(unit["contrasts"]["failure"], "-5000.000")
        record = self.call(
            "GET", CROSS_ROOT + "/crossover/units/ab-two/observations"
        )
        broken = next(item for item in record["cycles"] if item["cycle_id"] == "broken")
        self.assertEqual(broken["accepted_period"], 1)
        self.assertEqual(broken["state"], "aborted")

    def test_historical_schedule_replay_keeps_units_enrolled_after_that_position(self):
        # Enroll in two batches so the second batch carries later schedule seq rows.
        self.create_crossover()
        self.register_site()
        first = self.enroll_crossover([
            crossover_unit("ab-one", "ab"), crossover_unit("ba-one", "ba"),
        ])
        earliest_seq = 1
        second = self.enroll_crossover([
            crossover_unit("ab-two", "ab"), crossover_unit("ba-two", "ba"),
        ])
        self.open_crossover(second["version"])
        rows = self.balanced_events()
        cut = self.cross_deliver("batch", rows)["current_cut"]
        # Replaying at the earliest ledger position must not crash (KeyError) and
        # must keep every registered unit via its earliest known schedule.
        replay = self.cross_preview(cut=cut, schedule_seq=earliest_seq)
        unit_ids = {item["unit_id"] for item in replay["units"]}
        self.assertEqual(
            unit_ids, {"ab-one", "ab-two", "ba-one", "ba-two"}
        )
        market = replay["market_results"][0]
        for block in market["cells"][0]["sequences"]:
            self.assertEqual(block["paired_units"], 2)

    def test_markets_sharing_a_cell_key_do_not_contaminate_each_others_gates(self):
        plan_data = crossover_plan()
        plan_data["markets"] = [
            {"market_id": "alpine", "label": "Alpine",
             "cells": [{"climate": "cold", "load": "light", "weight_bp": 10_000}]},
            {"market_id": "boreal", "label": "Boreal",
             "cells": [{"climate": "cold", "load": "light", "weight_bp": 10_000}]},
        ]
        self.create_crossover(plan_data)
        self.register_site("north")
        self.call("POST", CROSS_ROOT + "/sites", {
            "site_id": "south", "market_id": "boreal",
            "climate": "cold", "timezone": "Europe/Berlin",
        })
        units = []
        for sequence in ("ab", "ba"):
            for number in (1, 2):
                units.append(crossover_unit(f"al-{sequence}-{number}", sequence,
                                            site_id="north"))
                units.append(crossover_unit(f"bo-{sequence}-{number}", sequence,
                                            site_id="south"))
        result = self.enroll_crossover(units)
        self.open_crossover(result["version"])
        rows = []
        # Alpine: clean balanced improvement in both sequences.
        for unit in (u for u in units if u["site_id"] == "north"):
            is_ab = unit["sequence"] == "ab"
            rows += cross_observations(
                unit["unit_id"], "p1", "2026-10-25T09:00:00+00:00",
                manual=is_ab, wh=100,
            )
            rows += cross_observations(
                unit["unit_id"], "p2", "2026-11-01T09:00:00+00:00",
                manual=not is_ab, wh=90,
            )
        # Boreal: AB improves fully, BA shows no manual saving -> divergence fail.
        for unit in (u for u in units if u["site_id"] == "south"):
            is_ab = unit["sequence"] == "ab"
            rows += cross_observations(
                unit["unit_id"], "p1", "2026-10-25T09:00:00+00:00",
                manual=True, wh=100,
            )
            rows += cross_observations(
                unit["unit_id"], "p2", "2026-11-01T09:00:00+00:00",
                manual=not is_ab, wh=90,
            )
        self.cross_deliver("batch", rows)
        preview = self.cross_preview()
        by_market = {item["market_id"]: item for item in preview["market_results"]}
        self.assertEqual(len(by_market["alpine"]["cells"]), 1)
        self.assertEqual(len(by_market["boreal"]["cells"]), 1)
        # Boreal must carry the divergence failure; alpine stays decidable/advance.
        self.assertIn(
            "sequence_divergence_manual_bp", by_market["boreal"]["blocked_by"]
        )
        self.assertNotIn(
            "sequence_divergence_manual_bp", by_market["alpine"]["blocked_by"]
        )
        self.assertEqual(by_market["alpine"]["recommendation"], "advance")
        self.assertEqual(by_market["boreal"]["recommendation"], "hold")
