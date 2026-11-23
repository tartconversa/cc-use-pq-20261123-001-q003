from tests.support import (
    CROSS_MAIN,
    CROSS_ROOT,
    CrossoverCase,
    cross_observations,
)


class CrossoverPeriodTest(CrossoverCase):
    def test_cycle_starting_in_period_but_closing_in_washout_is_not_accepted(self):
        self.setup_crossover()
        rows = self.balanced_events()
        # Starts 2026-10-28 22:00 UTC (29 Oct local Berlin) and closes 23:10 UTC,
        # entirely inside the first period local dates but the next cycle below
        # straddles the washout boundary.
        rows += cross_observations(
            "ab-one", "straddle", "2026-10-28T22:30:00+00:00", wh=40, seconds=7200
        )
        self.cross_deliver("batch", rows)
        record = self.call(
            "GET", CROSS_ROOT + "/crossover/units/ab-one/observations"
        )
        straddle = next(item for item in record["cycles"] if item["cycle_id"] == "straddle")
        self.assertIsNone(straddle["accepted_period"])
        self.assertEqual(straddle["outside_reason"], "crosses_period_boundary")
        self.assertEqual(straddle["nominal_period"], 1)

    def test_cycle_fully_inside_washout_is_excluded_even_when_start_date_alone_misleads(self):
        self.setup_crossover()
        rows = self.balanced_events()
        rows += cross_observations(
            "ab-one", "wash", "2026-10-30T10:00:00+00:00", wh=10
        )
        self.cross_deliver("batch", rows)
        record = self.call(
            "GET", CROSS_ROOT + "/crossover/units/ab-one/observations"
        )
        wash = next(item for item in record["cycles"] if item["cycle_id"] == "wash")
        self.assertIsNone(wash["accepted_period"])
        self.assertEqual(wash["outside_reason"], "washout_period")

    def test_transition_alarm_remains_visible_to_field_even_though_excluded(self):
        self.setup_crossover()
        rows = self.balanced_events()
        rows += cross_observations(
            "ab-one", "wash", "2026-10-30T10:00:00+00:00", wh=10, alarm=True
        )
        cut = self.cross_deliver("batch", rows)["current_cut"]
        queue = self.call(
            "GET", CROSS_ROOT + "/crossover/markets/alpine/inspection",
            params={"cut": cut},
        )
        categories = {item["category"] for item in queue["items"]}
        self.assertIn("transition_alarm", categories)
        entry = next(item for item in queue["items"] if item["category"] == "transition_alarm")
        self.assertEqual(entry["cycle_id"], "wash")
        self.assertEqual(entry["issues"], ["washout_period"])
        self.assertEqual(queue["counts"]["transition_alarm"], 1)
        market = self.cross_market()
        gate = next(g for g in market["gates"] if g["gate"] == "transition_alarms_visible")
        self.assertEqual(gate["state"], "fail")
        self.assertEqual(market["recommendation"], "hold")

    def test_period_acceptance_uses_full_cycle_not_start_date(self):
        self.setup_crossover()
        rows = self.balanced_events()
        # Starts inside period 1 (28 Oct local) but the two-hour run closes on
        # 29 Oct local which is the first washout date.
        rows += cross_observations(
            "ab-two", "edge", "2026-10-28T22:00:00+00:00", wh=30, seconds=7200
        )
        self.cross_deliver("batch", rows)
        preview = self.cross_preview()
        unit = next(item for item in preview["units"] if item["unit_id"] == "ab-two")
        edge = next(
            item for item in
            self.call("GET", CROSS_ROOT + "/crossover/units/ab-two/observations")["cycles"]
            if item["cycle_id"] == "edge"
        )
        self.assertIsNone(edge["accepted_period"])
        # The regular p1 cycle still keeps ab-two paired, so the edge cycle alone
        # must not inflate any period's usable count.
        self.assertEqual(unit["pairing"], "paired")

    def test_local_timezone_governs_period_assignment(self):
        # 21:30 UTC on 24 Oct is 23:30 local Berlin (UTC+2), still the same local
        # date; the assignment must follow the site zone rather than the UTC date.
        self.setup_crossover()
        rows = self.balanced_events()
        rows += cross_observations(
            "ba-one", "late", "2026-10-24T21:30:00+00:00", wh=15
        )
        self.cross_deliver("batch", rows)
        record = self.call(
            "GET", CROSS_ROOT + "/crossover/units/ba-one/observations"
        )
        late = next(item for item in record["cycles"] if item["cycle_id"] == "late")
        self.assertEqual(late["local_date"], "2026-10-24")
        self.assertEqual(late["accepted_period"], 1)
        self.assertEqual(late["nominal_period"], 1)
