"""Forecast arithmetic is offline and must not invent missing timing coverage."""
import unittest
from datetime import datetime, timedelta, timezone

import revision224 as study
from inspect_calibration import request_timing, runtime_forecast


class RuntimeForecastTests(unittest.TestCase):
    def test_complete_balanced_timing_and_completed_cells_are_not_counted_twice(self):
        config = study.load_config()
        rows = [{**c, 'request_roundtrip_seconds': 10} for c in study.calibration_cells(config)]
        completed = {r['run_id'] for r in rows}
        estimate = runtime_forecast(config, rows, completed)
        self.assertAlmostEqual(estimate['full_study_api_hours'], 8960 / 3600)
        self.assertAlmostEqual(estimate['remaining_api_hours'], 8280 / 3600)
        extra = next(c['run_id'] for c in study.cells(config) if c['run_id'] not in completed)
        estimate = runtime_forecast(config, rows, completed | {extra})
        self.assertAlmostEqual(estimate['remaining_api_hours'], 8270 / 3600)

    def test_missing_or_invalid_measurements_do_not_produce_a_forecast(self):
        config = study.load_config()
        rows = [{**c, 'request_roundtrip_seconds': 10} for c in study.calibration_cells(config)]
        self.assertIsNone(runtime_forecast(config, rows[:-1], set()))
        self.assertIsNone(runtime_forecast(config, rows + rows[:1], set()))
        rows[0]['request_roundtrip_seconds'] = None
        self.assertIsNone(runtime_forecast(config, rows, set()))

    def test_resume_downtime_is_not_api_time_and_backward_clock_is_rejected(self):
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        resumed = start + timedelta(days=1)
        timings = [(start, start + timedelta(seconds=10)), (resumed, resumed + timedelta(seconds=20))]
        self.assertEqual(request_timing(timings)['request_roundtrip_seconds'], 30)
        self.assertEqual(request_timing(timings)['elapsed_span_seconds'], 86420)
        with self.assertRaises(ValueError):
            request_timing([(resumed, start)])
        with self.assertRaises(ValueError):
            request_timing([])


if __name__ == '__main__':
    unittest.main()
