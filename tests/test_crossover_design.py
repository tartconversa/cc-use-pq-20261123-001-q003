from tests.support import (
    CROSS_ROOT,
    CrossoverCase,
    crossover_plan,
    crossover_unit,
)


class CrossoverDesignTest(CrossoverCase):
    def test_open_requires_both_sequences_in_every_cell_and_reports_gaps(self):
        self.create_crossover()
        self.register_site()
        result = self.enroll_crossover([
            crossover_unit("ab-one", "ab"), crossover_unit("ab-two", "ab"),
            crossover_unit("ba-one", "ba"),
        ])
        checks = self.call("GET", CROSS_ROOT + "/crossover/open-checks")
        self.assertFalse(checks["ready"])
        self.assertEqual(checks["allocation_gaps"], [{
            "market": "alpine", "cell": "cold/light", "sequence": "ba",
            "enrolled": 1, "required": 2,
        }])
        failure = self.call("POST", CROSS_ROOT + "/crossover/open",
                            {"expected_version": result["version"]}, status=409)
        self.assertEqual(failure["code"], "crossover_allocation_shortfall")
        self.assertEqual(self.call("GET", CROSS_ROOT)["phase"], "draft")

    def test_schedule_must_hold_two_periods_and_minimum_washout(self):
        self.create_crossover(crossover_plan(washout_days=3))
        self.register_site()
        bad = crossover_unit(
            "ab-one", "ab", first_until="2026-10-30", second_from="2026-10-31"
        )
        problem = self.call(
            "POST", CROSS_ROOT + "/crossover/enrollment", {"units": [bad]}, status=422
        )
        self.assertEqual(problem["code"], "washout_too_short")
        self.assertEqual(problem["details"], {"observed": 1, "required": 3})

    def test_schedule_report_lists_local_periods_and_washout(self):
        self.setup_crossover()
        report = self.call("GET", CROSS_ROOT + "/crossover/schedule")
        first = next(item for item in report["units"] if item["unit_id"] == "ab-one")
        self.assertEqual(first["washout_days"], 2)
        self.assertTrue(first["washout_ok"])
        self.assertEqual(first["cell"], "cold/light")
        self.assertEqual(first["sequence"], "ab")

    def test_parallel_and_crossover_endpoints_reject_each_others_trials(self):
        self.setup_crossover()
        guard = self.call("GET", CROSS_ROOT + "/roster", status=409)
        self.assertEqual(guard["code"], "not_parallel")
        guard = self.call("GET", CROSS_ROOT + "/preview", status=409)
        self.assertEqual(guard["code"], "not_parallel")
        # The parallel enrollment path is rejected by design type even while open.
        from tests.support import enrollment

        blocked = self.call(
            "POST", CROSS_ROOT + "/enrollment",
            {"units": [enrollment("sneaky", "standard")]}, status=409,
        )
        self.assertEqual(blocked["code"], "not_parallel")

    def test_schedule_correction_appends_revision_and_keeps_history(self):
        opened = self.setup_crossover()
        history_before = self.call(
            "GET", CROSS_ROOT + "/crossover/units/ab-one/schedule"
        )["revisions"]
        self.assertEqual([row["revision"] for row in history_before], [1])
        correction = self.call(
            "PUT", CROSS_ROOT + "/crossover/units/ab-one/schedule",
            {
                "expected_revision": 1,
                "first_from": "2026-10-24", "first_until": "2026-10-28",
                "second_from": "2026-10-30", "second_until": "2026-11-05",
                "reason": "site-diary-fix",
            },
        )
        self.assertEqual(correction["schedule_revision"], 2)
        self.assertGreaterEqual(correction["schedule_seq"], 1)
        history_after = self.call(
            "GET", CROSS_ROOT + "/crossover/units/ab-one/schedule"
        )["revisions"]
        self.assertEqual([row["revision"] for row in history_after], [1, 2])
        self.assertEqual(history_after[1]["reason"], "site-diary-fix")
        # A stale expected revision is rejected.
        stale = self.call(
            "PUT", CROSS_ROOT + "/crossover/units/ab-one/schedule",
            {
                "expected_revision": 1,
                "first_from": "2026-10-24", "first_until": "2026-10-28",
                "second_from": "2026-10-30", "second_until": "2026-11-05",
                "reason": "again",
            },
            status=409,
        )
        self.assertEqual(stale["code"], "schedule_revision")
        # Open-state corrections do not move the trial version.
        self.assertEqual(self.call("GET", CROSS_ROOT)["version"], opened["version"])

    def test_correction_cannot_shorten_washout_below_policy(self):
        self.setup_crossover()
        problem = self.call(
            "PUT", CROSS_ROOT + "/crossover/units/ab-one/schedule",
            {
                "expected_revision": 1,
                "first_from": "2026-10-24", "first_until": "2026-10-30",
                "second_from": "2026-10-31", "second_until": "2026-11-05",
                "reason": "bad",
            },
            status=422,
        )
        self.assertEqual(problem["code"], "washout_too_short")

    def test_repeated_enrollment_is_idempotent(self):
        self.create_crossover()
        self.register_site()
        units = [crossover_unit("ab-one", "ab")]
        first = self.enroll_crossover(units)
        second = self.enroll_crossover(units)
        self.assertEqual(first["added"], 1)
        self.assertEqual(second["added"], 0)
        self.assertEqual(second["version"], first["version"])
