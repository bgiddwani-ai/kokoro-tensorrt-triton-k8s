import csv
import json
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2] / "results"


def read_csv(name):
    with (ROOT / name).open() as stream:
        return list(csv.DictReader(stream))


class ReferenceTests(unittest.TestCase):
    def test_summaries_match_raw_measurements(self):
        summary = json.loads((ROOT / "summary.json").read_text())
        self.assertFalse(summary["context"]["kubernetes_measured"])
        models = read_csv("model_samples.csv")
        for backend, groups in summary["core_p50_ms"].items():
            for group, expected in groups.items():
                samples = [
                    float(row["latency_ms"])
                    for row in models
                    if row["backend"] == backend and row["length"] == group
                ]
                self.assertEqual(len(samples), 32)
                self.assertAlmostEqual(float(np.median(samples)), expected)
        sustained = read_csv("sustained_samples.csv")
        measured = summary["sustained"]
        self.assertEqual(len(sustained), measured["requests"])
        self.assertTrue(all(row["finite"] == "True" for row in sustained))
        self.assertAlmostEqual(
            len(sustained) / measured["duration_s"], measured["requests_per_s"]
        )
        for name, q in (
            ("p50_ms", 50),
            ("p95_ms", 95),
            ("p99_ms", 99),
            ("max_ms", 100),
        ):
            self.assertAlmostEqual(
                float(
                    np.percentile([float(row["latency_ms"]) for row in sustained], q)
                ),
                measured[name],
            )

    def test_full_historical_matrix_has_matching_samples(self):
        matrix, samples = read_csv("matrix.csv"), read_csv("matrix_samples.csv")
        self.assertEqual(len(matrix), 66)
        self.assertEqual(len(samples), 10560)
        for cell in matrix:
            subset = [
                row
                for row in samples
                if (row["config"], row["concurrency"])
                == (cell["config"], cell["concurrency"])
            ]
            self.assertEqual(len(subset), int(cell["requests"]))
            self.assertAlmostEqual(
                len(subset) / float(cell["elapsed_s"]), float(cell["requests_per_s"])
            )
