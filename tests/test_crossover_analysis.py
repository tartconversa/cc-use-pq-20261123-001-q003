from tests.support import (
    CROSS_ROOT,
    CrossoverCase,
    cross_observations,
    crossover_plan,
    crossover_unit,
)


def seq_block(market, cell, sequence):
    return next(
        item for item in market["cells"][0]["sequences"] if item["sequence"] == sequence
    )


class CrossoverAnalysisTest(CrossoverCase):
    def test_within_unit_effect_contrasts_assisted_and_standard_periods(self):
        self.setup_crossover()
        self.cross_deliver("batch", self.balanced_events())
        market = self.cross_market()
        ab = seq_block(market, None, "ab")
        ba = seq_block(market, None, "ba")
        # Every standard period has a manual cycle, every assisted period has none.
        self.assertEqual(ab["effects"]["manual"], "-10000.000")
        self.assertEqual(ba["effects"]["manual"], "-10000.000")
        self.assertEqual(ab["periods"]["1"]["arm"], "standard")
        self.assertEqual(ab["periods"]["2"]["arm"], "assisted")
        self.assertEqual(ba["periods"]["1"]["arm"], "assisted")
        self.assertEqual(ba["periods"]["2"]["arm"], "standard")
        self.assertEqual(market["cells"][0]["merged_effects"]["manual"], "-10000.000")

    def test_market_judgement_uses_only_units_complete_and_eligible_in_both_periods(self):
        self.setup_crossover()
        rows = self.balanced_events()
        # ab-two loses its second period: replace the p2 cycle with a washout cycle.
        rows = [r for r in rows if not (r["unit_id"] == "ab-two" and r["cycle_id"] == "p2")]
        rows += cross_observations(
            "ab-two", "gone", "2026-10-30T10:00:00+00:00", wh=10
        )
        self.cross_deliver("batch", rows)
        preview = self.cross_preview()
        unit = next(item for item in preview["units"] if item["unit_id"] == "ab-two")
        self.assertEqual(unit["pairing"], "period1_only")
        market = preview["market_results"][0]
        ab = seq_block(market, None, "ab")
        # ab-two stays in the attrition/coverage counts but contributes no effect.
        self.assertEqual(ab["recruited_units"], 2)
        self.assertEqual(ab["paired_units"], 1)
        self.assertEqual(ab["period1_only_units"], 1)
        coverage = self.call(
            "GET", CROSS_ROOT + "/crossover/markets/alpine/coverage"
        )
        ab_coverage = next(
            item for item in coverage["cells"][0]["sequences"] if item["sequence"] == "ab"
        )
        self.assertEqual(ab_coverage["pairing_coverage_bp"], "5000.000")
        # One sequence is now below the two paired units minimum.
        self.assertTrue(any(g["reason"] == "paired_units" for g in market["cells"][0]["gaps"]))
        self.assertEqual(market["recommendation"], "insufficient")

    def test_missing_sequence_cannot_be_replaced_by_the_other(self):
        plan_data = crossover_plan(required_per_sequence=1)
        self.create_crossover(plan_data)
        self.register_site()
        units = [
            crossover_unit("ab-one", "ab"), crossover_unit("ab-two", "ab"),
            crossover_unit("ba-one", "ba"), crossover_unit("ba-two", "ba"),
        ]
        result = self.enroll_crossover(units)
        self.open_crossover(result["version"])
        rows = self.balanced_events()
        # Remove every BA observation so the whole sequence has no pairing.
        rows = [r for r in rows if r["unit_id"] not in {"ba-one", "ba-two"}]
        self.cross_deliver("batch", rows)
        market = self.cross_market()
        ba = seq_block(market, None, "ba")
        self.assertEqual(ba["paired_units"], 0)
        self.assertIsNone(ba["effects"]["manual"])
        self.assertEqual(ba["recruited_units"], 2)
        self.assertEqual(market["recommendation"], "insufficient")
        gate = next(
            g for g in market["gates"] if g["gate"] == "crossover_sequence_coverage"
        )
        self.assertEqual(gate["state"], "unavailable")

    def test_sequence_divergence_holds_an_otherwise_favourable_merged_result(self):
        # The AB sequence sees a large manual saving; BA barely improves. The
        # merged mean is favourable, yet divergence exceeds the registered limit.
        plan_data = crossover_plan(divergence_bp=2_000)
        self.setup_crossover(plan_data=plan_data)
        rows = []
        for unit in ("ab-one", "ab-two"):
            rows += cross_observations(unit, "p1", "2026-10-25T09:00:00+00:00",
                                       manual=True, wh=100)
            rows += cross_observations(unit, "p2", "2026-11-01T09:00:00+00:00",
                                       manual=False, wh=90)
        for unit in ("ba-one", "ba-two"):
            # Both BA periods carry manual action: zero within-unit improvement.
            rows += cross_observations(unit, "p1", "2026-10-25T09:00:00+00:00",
                                       manual=True, wh=90)
            rows += cross_observations(unit, "p2", "2026-11-01T09:00:00+00:00",
                                       manual=True, wh=100)
        self.cross_deliver("batch", rows)
        market = self.cross_market()
        divergence = next(
            g for g in market["gates"]
            if g["gate"] == "sequence_divergence_manual_bp"
        )
        self.assertEqual(divergence["state"], "fail")
        self.assertEqual(divergence["observed"], "10000.000")
        self.assertEqual(divergence["boundary"], 2_000)
        self.assertEqual(market["recommendation"], "hold")
        # Operators can trace which round/condition produced the split.
        ab = seq_block(market, None, "ab")
        ba = seq_block(market, None, "ba")
        self.assertEqual(ab["effects"]["manual"], "-10000.000")
        self.assertEqual(ba["effects"]["manual"], "0.000")

    def test_balanced_compliant_trial_advances_and_reports_per_sequence_samples(self):
        self.setup_crossover()
        self.cross_deliver("batch", self.balanced_events())
        market = self.cross_market()
        self.assertEqual(market["recommendation"], "advance", market["blocked_by"])
        for sequence in ("ab", "ba"):
            block = seq_block(market, None, sequence)
            self.assertEqual(block["paired_units"], 2)
            self.assertEqual(block["recruited_units"], 2)
            for period in block["periods"].values():
                self.assertEqual(period["sample_units"], 2)

    def test_energy_divergence_gate_uses_watthours_not_basis_points(self):
        plan_data = crossover_plan(divergence_wh=5)
        self.setup_crossover(plan_data=plan_data)
        rows = []
        # AB assisted period uses 90 and 70 Wh (mean 80); BA assisted period uses
        # 90 Wh twice. Standard periods are 100 Wh everywhere.
        for unit, assisted_wh in (("ab-one", 90), ("ab-two", 70)):
            rows += cross_observations(unit, "p1", "2026-10-25T09:00:00+00:00",
                                       manual=True, wh=100)
            rows += cross_observations(unit, "p2", "2026-11-01T09:00:00+00:00",
                                       wh=assisted_wh)
        for unit in ("ba-one", "ba-two"):
            rows += cross_observations(unit, "p1", "2026-10-25T09:00:00+00:00", wh=90)
            rows += cross_observations(unit, "p2", "2026-11-01T09:00:00+00:00",
                                       manual=True, wh=100)
        self.cross_deliver("batch", rows)
        market = self.cross_market()
        gate = next(
            g for g in market["gates"] if g["gate"] == "sequence_divergence_energy_wh"
        )
        # AB energy effect = 80 - 100 = -20 Wh; BA = 90 - 100 = -10 Wh; spread 10.
        self.assertEqual(gate["state"], "fail")
        self.assertEqual(gate["observed"], "10.000")
        self.assertEqual(gate["boundary"], 5)

    def test_one_period_units_remain_in_attrition_counts(self):
        self.setup_crossover()
        rows = self.balanced_events()
        # ba-one never observes a second period cycle.
        rows = [r for r in rows if not (r["unit_id"] == "ba-one" and r["cycle_id"] == "p2")]
        self.cross_deliver("batch", rows)
        coverage = self.call(
            "GET", CROSS_ROOT + "/crossover/markets/alpine/coverage"
        )
        ba = next(
            item for item in coverage["cells"][0]["sequences"] if item["sequence"] == "ba"
        )
        self.assertEqual(ba["recruited_units"], 2)
        self.assertEqual(ba["period2_only_units"], 0)
        self.assertEqual(ba["period1_only_units"], 1)
        self.assertEqual(ba["paired_units"], 1)
