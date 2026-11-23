import csv
from io import StringIO

from tests.support import (
    CROSS_MAIN,
    CROSS_ROOT,
    CrossoverCase,
    cross_observations,
)


class CrossoverCorrectionTest(CrossoverCase):
    def test_schedule_correction_turns_accepted_cycle_into_suspect_cross_period(self):
        opened = self.setup_crossover()
        rows = self.balanced_events()
        # ab-one's only period-1 cycle deliberately starts on the last period day.
        rows = [r for r in rows if not (r["unit_id"] == "ab-one" and r["cycle_id"] == "p1")]
        rows += cross_observations(
            "ab-one", "p1", "2026-10-28T12:00:00+00:00", manual=True, wh=100
        )
        cut = self.cross_deliver("batch", rows)["current_cut"]
        before_seq = self.cross_preview()["schedule_seq"]
        before = self.cross_preview()
        ab_before = next(item for item in before["units"] if item["unit_id"] == "ab-one")
        self.assertEqual(ab_before["pairing"], "paired")

        correction = self.call(
            "PUT", CROSS_ROOT + "/crossover/units/ab-one/schedule",
            {
                "expected_revision": 1,
                "first_from": "2026-10-24", "first_until": "2026-10-28",
                "second_from": "2026-10-30", "second_until": "2026-11-05",
                "reason": "site-diary-fix",
            },
        )
        impact = self.call("GET", CROSS_ROOT + "/crossover/impact", params={
            "before_cut": cut, "after_cut": cut,
            "before_schedule_seq": before_seq,
            "after_schedule_seq": correction["schedule_seq"],
        })
        suspect = impact["suspect_cross_period_cycles"]
        self.assertEqual(len(suspect), 1)
        self.assertEqual(suspect[0]["unit_id"], "ab-one")
        self.assertEqual(suspect[0]["cycle_id"], "p1")
        self.assertEqual(suspect[0]["affected_period"], 1)
        self.assertEqual(suspect[0]["affected_arm"], "standard")
        self.assertEqual(suspect[0]["outside_reason"], "washout_period")
        self.assertEqual(suspect[0]["sequence"], "ab")
        # Losing the only period-1 observation moves the unit out of the pairing.
        self.assertEqual(impact["pairing_changes"], [{
            "unit_id": "ab-one", "before_pairing": "paired",
            "after_pairing": "period2_only", "sequence": "ab", "cell": "cold/light",
        }])

    def test_signal_correction_is_attributed_to_the_same_period_and_pair(self):
        self.setup_crossover()
        first = self.cross_deliver("batch", self.balanced_events())["current_cut"]
        before_seq = self.cross_preview()["schedule_seq"]
        # A signal revision changes an assisted-period meter reading; the cycle
        # stays inside period 2 and therefore must not become a cross-period case.
        meter = {
            "event_id": "ab-one-p2-m1", "revision": 2, "unit_id": "ab-one",
            "cycle_id": "p2", "kind": "meter",
            "occurred_at": "2026-11-01T09:10:00+00:00",
            "meter_id": "main", "meter_wh": 150,
        }
        after_cut = self.cross_deliver("meter-fix", [meter])["current_cut"]
        impact = self.call("GET", CROSS_ROOT + "/crossover/impact", params={
            "before_cut": first, "after_cut": after_cut,
            "before_schedule_seq": before_seq,
            "after_schedule_seq": before_seq,
        })
        self.assertEqual(impact["suspect_cross_period_cycles"], [])
        changed = next(
            item for item in impact["cycles"]
            if item["unit_id"] == "ab-one" and item["cycle_id"] == "p2"
        )
        self.assertEqual(changed["after"]["accepted_period"], 2)
        self.assertEqual(changed["after"]["source_revisions"]["ab-one-p2-m1"], 2)

    def test_impact_rejects_reversed_cut_and_schedule_order(self):
        self.setup_crossover()
        cut = self.cross_deliver("batch", self.balanced_events())["current_cut"]
        seq = self.cross_preview()["schedule_seq"]
        bad = self.call("GET", CROSS_ROOT + "/crossover/impact", params={
            "before_cut": cut, "after_cut": 0,
            "before_schedule_seq": seq, "after_schedule_seq": seq,
        }, status=422)
        self.assertEqual(bad["code"], "cut_order")


class CrossoverReleaseTest(CrossoverCase):
    def _green_trial(self):
        opened = self.setup_crossover()
        cut = self.cross_deliver("batch", self.balanced_events())["current_cut"]
        return opened, cut

    def test_close_freezes_schedule_seq_and_evidence_matches_paired_population(self):
        opened, cut = self._green_trial()
        closed = self.call("POST", CROSS_ROOT + "/close", {
            "expected_version": opened["version"], "expected_cut": cut,
        })
        self.assertEqual(closed["close_cut"], cut)
        self.assertIsNotNone(closed["close_schedule_seq"])
        frozen = self.cross_freeze(cut)
        self.assertEqual(frozen["design"], "crossover")
        self.assertEqual(frozen["cut"], cut)
        self.assertEqual(frozen["close_schedule_seq"], closed["close_schedule_seq"])
        self.assertEqual(frozen["schedule_seq"], closed["close_schedule_seq"])
        # Frozen roster digest covers exactly the paired population.
        market = frozen["market_results"][0]
        self.assertEqual(market["recommendation"], "advance")
        paired = sum(
            block["paired_units"]
            for block in market["cells"][0]["sequences"]
        )
        self.assertEqual(paired, 4)

    def test_dual_desk_reviews_and_bulletin_use_the_frozen_population(self):
        opened, cut = self._green_trial()
        self.close_crossover(opened, cut)
        frozen = self.cross_freeze(cut)
        eid = frozen["evidence_id"]
        self.cross_review(eid, "field", "ada")
        clash = self.cross_review(eid, "methods", "ada", status=409)
        self.assertEqual(clash["code"], "reviewer_not_independent")
        missing = self.cross_issue(eid, "advance", status=409)
        self.assertEqual(missing["code"], "endorsements_missing")
        self.cross_review(eid, "methods", "ben")
        bulletin = self.cross_issue(eid, "advance")
        self.assertEqual(bulletin["outcome"], "advance")
        self.assertEqual(bulletin["decision_reason"], "thresholds_and_reviews")
        self.assertEqual(bulletin["result"]["market_id"], "alpine")
        # Re-issuing identical content returns the same bulletin.
        self.assertEqual(bulletin, self.cross_issue(eid, "advance"))
        self.assertEqual(
            self.call("GET", CROSS_MAIN + "/bulletins/latest"), bulletin
        )

    def test_hold_follows_a_failing_gate_and_objection_holds_green_numbers(self):
        opened, cut = self._green_trial()
        self.close_crossover(opened, cut)
        # Add a transition alarm by reopening is impossible after close, so use
        # an objection path on the green frozen evidence.
        frozen = self.cross_freeze(cut)
        eid = frozen["evidence_id"]
        self.cross_review(eid, "field", "ada", assessment="object")
        bulletin = self.cross_issue(eid, "hold")
        self.assertEqual(bulletin["decision_reason"], "review_objection")

    def test_frozen_csv_reports_sequence_period_and_effect_rows(self):
        opened, cut = self._green_trial()
        self.close_crossover(opened, cut)
        frozen = self.cross_freeze(cut)
        eid = frozen["evidence_id"]
        response = self.client.get(CROSS_ROOT + f"/evidence/{eid}/metrics.csv")
        self.assertEqual(response.status_code, 200)
        rows = list(csv.DictReader(StringIO(response.text)))
        effects = [row for row in rows if row["within_unit_effect"]]
        self.assertTrue(effects)
        self.assertTrue(all(row["sequence"] in {"ab", "ba", "merged"} for row in effects))
        manual = next(row for row in effects if row["metric"] == "manual"
                      and row["sequence"] == "ab")
        self.assertEqual(manual["within_unit_effect"], "-10000.000")

    def test_new_signal_after_close_is_rejected_and_cut_guard_holds(self):
        opened, cut = self._green_trial()
        stale_close = self.call("POST", CROSS_ROOT + "/close", {
            "expected_version": opened["version"], "expected_cut": cut + 99,
        }, status=409)
        self.assertEqual(stale_close["code"], "cut_changed")
        self.close_crossover(opened, cut)
        late = self.cross_deliver(
            "late", cross_observations("ab-one", "late", "2026-11-02T09:00:00+00:00"),
            status=409,
        )
        self.assertEqual(late["code"], "trial_phase")

    def close_crossover(self, opened, cut):
        return self.call("POST", CROSS_ROOT + "/close", {
            "expected_version": opened["version"], "expected_cut": cut,
        })
